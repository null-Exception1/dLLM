package main

import (
	"context"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	pb "dllm-cluster/data-plane/proto"
	"dllm-cluster/data-plane/ring"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const colabCoprocessorAddr = "edzik-34-124-254-109.run.pinggy-free.link:35913"

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

	targetAddr := "localhost:" + targetPort
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
		log.Printf("[Mesh Node :%s] 🎯 LOCAL COORDINATE HIT -> Layer: %d. Proxying to shared Colab GPU...", s.CurrentPort, layerIndex)

		req.TaskSerialNumber = uint32(intCastPort(s.CurrentPort))

		ctxTimeout, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		resp, err := s.colabClient.StreamLayerActivations(ctxTimeout, req)
		if err != nil {
			return nil, err
		}

		if resp.GeneratedToken != "" {
			log.Printf("🎉 [RESULT NODE MATCH] Mesh Node :%s successfully intercepted decoded token text from cloud: \"%s\"",
				s.CurrentPort, resp.GeneratedToken)
		}

		return resp, nil
	}

	log.Printf("[Mesh Node :%s] 🔀 RING TOPOLOGY SHIFT -> Layer: %d maps clockwise to node :%s. Forwarding...",
		s.CurrentPort, layerIndex, targetNodePort)

	peerClient, err := s.getPeerClient(targetNodePort)
	if err != nil {
		return &pb.StreamAck{Success: false, ErrorMessage: err.Error()}, nil
	}

	resp, err := peerClient.StreamLayerActivations(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.GeneratedToken != "" && s.CurrentPort == "50051" {
		log.Printf("🎉 [MESH INTERCEPT] Entry Shard :%s intercepted returned token path: \"%s\"",
			s.CurrentPort, resp.GeneratedToken)
	}

	return resp, nil
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

func main() {
	portFlag := flag.String("port", "50051", "The data-plane port for this node")
	flag.Parse()

	log.Printf("🛰️ Dialing cleartext TCP socket via Pinggy out to: %s", colabCoprocessorAddr)

	conn, err := grpc.NewClient(colabCoprocessorAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ Fatal: Failed to open transport pipeline to Colab coprocessor: %v", err)
	}
	defer conn.Close()

	log.Println("🎉 Success! Connected cleanly to the centralized Colab GPU coprocessor.")

	clusterRing := ring.NewHashRing(256)
	clusterRing.AddNode("50051")
	clusterRing.AddNode("50052")
	clusterRing.AddNode("50053")

	port := ":" + *portFlag
	listener, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("❌ Fatal: Failed to bind TCP socket: %v", err)
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

	go func() {
		log.Printf("🚀 dLLM Data-Plane Shard Node is listening on port %s...", port)
		if err := grpcServer.Serve(listener); err != nil {
			log.Fatalf("❌ Fatal: Server crashed: %v", err)
		}
	}()

	stopSignal := make(chan os.Signal, 1)
	signal.Notify(stopSignal, os.Interrupt, syscall.SIGTERM)
	<-stopSignal

	log.Printf("\n🧼 Clean Shutdown: Stopping Shard Node :%s...", *portFlag)
	grpcServer.GracefulStop()
}
