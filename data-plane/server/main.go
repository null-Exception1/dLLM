package main

import (
	"context"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	pb "dllm-cluster/data-plane/proto"
	"dllm-cluster/data-plane/ring"

	"google.golang.org/grpc"
)

type DataPlaneServer struct {
	pb.UnimplementedActivationStreamServer
	Ring        *ring.HashRing
	CurrentPort string
}

func (s *DataPlaneServer) StreamLayerActivations(ctx context.Context, req *pb.ActivationPayload) (*pb.StreamAck, error) {
	promptHash := req.GetPromptHash()
	taskSerial := req.GetTaskSerialNumber()
	layerIndex := req.GetLayerIndex()

	targetTargetNode := s.Ring.GetNode(promptHash, layerIndex, taskSerial)

	log.Printf("[Mesh Ring] 📥 Ingested Frame -> Prompt: 0x%X | Target Layer: %d", promptHash, layerIndex)

	if targetTargetNode == s.CurrentPort {
		log.Printf("            ROUTING HIT: Target node %s is LOCAL. Executing in-memory task loops.", targetTargetNode)

	} else {
		log.Printf("            ROUTING DEFLECTION: Task maps to node %s. Forwarding down the network wire.", targetTargetNode)
	}

	return &pb.StreamAck{Success: true}, nil
}

func main() {
	portFlag := flag.String("port", "50051", "The data-plane port for this node")
	flag.Parse()

	clusterRing := ring.NewHashRing(256)

	clusterRing.AddNode("50051")
	clusterRing.AddNode("50052")
	clusterRing.AddNode("50053")

	port := ":" + *portFlag
	listener, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf(" Fatal: Failed to bind TCP socket connection on port %s: %v", port, err)
	}

	grpcServer := grpc.NewServer()
	serverInstance := &DataPlaneServer{
		Ring:        clusterRing,
		CurrentPort: *portFlag,
	}

	pb.RegisterActivationStreamServer(grpcServer, serverInstance)

	go func() {
		log.Printf(" dLLM Data-Plane Network Node is listening on port %s...", port)
		if err := grpcServer.Serve(listener); err != nil {
			log.Fatalf(" Fatal: Data-Plane server crashed: %v", err)
		}
	}()

	stopSignal := make(chan os.Signal, 1)
	signal.Notify(stopSignal, os.Interrupt, syscall.SIGTERM)
	<-stopSignal

	log.Println("\n Stopping dLLM Data-Plane Server Node...")
	grpcServer.GracefulStop()
	log.Println(" Cluster connections closed cleanly.")
}
