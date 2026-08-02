package main

import (
	"bufio"
	"context"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	pb "dllm-cluster/data-plane/proto"
	"dllm-cluster/data-plane/ring"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type DataPlaneServer struct {
	pb.UnimplementedActivationStreamServer
	Ring        *ring.HashRing
	CurrentPort string
	colabClient pb.ActivationStreamClient
	colabConn   *grpc.ClientConn
	peerPool    map[string]pb.ActivationStreamClient
	poolMutex   sync.RWMutex
}

func (s *DataPlaneServer) getPeerClient(targetPort string) (pb.ActivationStreamClient, error) {
	s.poolMutex.RLock()
	client, exists := s.peerPool[targetPort]
	s.poolMutex.RUnlock()
	if exists {
		return client, nil
	}

	s.poolMutex.Lock()
	defer s.poolMutex.Unlock()
	if client, exists := s.peerPool[targetPort]; exists {
		return client, nil
	}

	targetAddr := "127.0.0.1:" + targetPort
	conn, err := grpc.Dial(targetAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	newClient := pb.NewActivationStreamClient(conn)
	s.peerPool[targetPort] = newClient
	return newClient, nil
}

func (s *DataPlaneServer) StreamLayerActivations(ctx context.Context, req *pb.ActivationPayload) (*pb.StreamAck, error) {
	promptHash := req.GetPromptHash()
	taskSerial := req.GetTaskSerialNumber()
	layerIndex := req.GetLayerIndex()

	targetNodePort := s.Ring.GetNode(promptHash, layerIndex, taskSerial)

	if targetNodePort == s.CurrentPort {
		log.Printf("[Mesh Node :%s] 🎯 COORDINATE HIT -> Processing Layer: %d", s.CurrentPort, layerIndex)

		req.TaskSerialNumber = uint32(intCastPort(s.CurrentPort))

		ctxTimeout, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		resp, err := s.colabClient.StreamLayerActivations(ctxTimeout, req)
		if err != nil {
			log.Printf("❌ Cloud Link Drop on Layer %d: %v", layerIndex, err)
			return &pb.StreamAck{Success: false, ErrorMessage: err.Error()}, nil
		}

		if layerIndex == 27 {
			log.Printf("🏆 [TERMINAL GATEWAY REACHED] Discovered Token Text: \"%s\"", resp.GeneratedToken)
			return resp, nil
		}

		// AUTONOMOUS LAYER AUTO-ADVANCE PIPELINE
		req.LayerIndex = layerIndex + 1
		log.Printf("[Mesh Node :%s] 🏎️ AUTO-ADVANCE -> Layer %d complete. Chaining forward to Layer %d...", s.CurrentPort, layerIndex, req.LayerIndex)

		nextTargetPort := s.Ring.GetNode(promptHash, req.LayerIndex, taskSerial)
		nextPeerClient, err := s.getPeerClient(nextTargetPort)
		if err != nil {
			return &pb.StreamAck{Success: false, ErrorMessage: err.Error()}, nil
		}

		return nextPeerClient.StreamLayerActivations(ctx, req)
	}

	log.Printf("[Mesh Node :%s] 🔀 RING TOPOLOGY SHIFT -> Layer: %d maps clockwise to port :%s. Forwarding...", s.CurrentPort, layerIndex, targetNodePort)
	peerClient, err := s.getPeerClient(targetNodePort)
	if err != nil {
		return &pb.StreamAck{Success: false, ErrorMessage: err.Error()}, nil
	}
	return peerClient.StreamLayerActivations(ctx, req)
}

func intCastPort(p string) int {
	if p == "50051" {
		return 1
	}
	if p == "50052" {
		return 2
	}
	return 3
}

func readEnvAddress(filePath string) string {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("❌ Config Error: %v", err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 && parts[0] == "COLAB_COPROCESSOR_ADDR" {
			return strings.Trim(parts[1], "\"")
		}
	}
	return ""
}

func main() {
	portFlag := flag.String("port", "50051", "The data-plane port for this node")
	flag.Parse()

	colabCoprocessorAddr := readEnvAddress(".env")
	log.Printf("🛰️ [Config Hot-Reload] Loaded active remote gateway address: %s", colabCoprocessorAddr)

	conn, err := grpc.Dial(colabCoprocessorAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ Fatal: %v", err)
	}

	clusterRing := ring.NewHashRing(256)
	clusterRing.AddNode("50051")
	clusterRing.AddNode("50052")
	clusterRing.AddNode("50053")

	port := ":" + *portFlag
	listener, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("❌ Fatal: %v", err)
	}

	grpcServer := grpc.NewServer()
	serverInstance := &DataPlaneServer{
		Ring:        clusterRing,
		CurrentPort: *portFlag,
		colabClient: pb.NewActivationStreamClient(conn),
		colabConn:   conn,
		peerPool:    make(map[string]pb.ActivationStreamClient),
	}

	pb.RegisterActivationStreamServer(grpcServer, serverInstance)
	go func() { _ = grpcServer.Serve(listener) }()

	stopSignal := make(chan os.Signal, 1)
	signal.Notify(stopSignal, os.Interrupt, syscall.SIGTERM)
	<-stopSignal
	conn.Close()
	grpcServer.GracefulStop()
}
