package ring

import (
	"fmt"
	"hash/crc32"
	"sort"
	"strconv"
	"sync"
)

type HashRing struct {
	virtualNodes map[uint32]string
	sortedKeys   []uint32 // A sorted slice of all active virtual positions on the wheel
	virtualRatio int
	mutex        sync.RWMutex
}

func NewHashRing(virtualRatio int) *HashRing {
	return &HashRing{
		virtualNodes: make(map[uint32]string),
		sortedKeys:   []uint32{},
		virtualRatio: virtualRatio,
	}
}

func (r *HashRing) hashKey(key string) uint32 {
	return crc32.ChecksumIEEE([]byte(key))
}

func (r *HashRing) AddNode(node string) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	for i := 0; i < r.virtualRatio; i++ {
		virtualKey := node + "#" + strconv.Itoa(i)
		wheelAddress := r.hashKey(virtualKey)
		r.virtualNodes[wheelAddress] = node
		r.sortedKeys = append(r.sortedKeys, wheelAddress)
	}

	sort.Slice(r.sortedKeys, func(i, j int) bool {
		return r.sortedKeys[i] < r.sortedKeys[j]
	})
}

func (r *HashRing) RemoveNode(node string) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	for i := 0; i < r.virtualRatio; i++ {
		virtualKey := node + "#" + strconv.Itoa(i)
		wheelAddress := r.hashKey(virtualKey)
		delete(r.virtualNodes, wheelAddress)
	}

	// Rebuild the sorted key slice to seal up layout gaps
	var updatedKeys []uint32
	for k := range r.virtualNodes {
		updatedKeys = append(updatedKeys, k)
	}
	sort.Slice(updatedKeys, func(i, j int) bool {
		return updatedKeys[i] < updatedKeys[j]
	})
	r.sortedKeys = updatedKeys
}

func (r *HashRing) GetNode(promptHash uint64, layerIndex uint32, taskSerial uint32) string {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	if len(r.sortedKeys) == 0 {
		return ""
	}

	compoundToken := fmt.Sprintf("%d-%d-%d", promptHash, layerIndex, taskSerial)
	wheelAddress := r.hashKey(compoundToken)

	idx := sort.Search(len(r.sortedKeys), func(i int) bool {
		return r.sortedKeys[i] >= wheelAddress
	})

	if idx == len(r.sortedKeys) {
		idx = 0
	}

	return r.virtualNodes[r.sortedKeys[idx]]
}
