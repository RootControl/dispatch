package tiers

import "math"

// kmeans partitions unit vectors into k clusters by cosine similarity
// (spherical k-means). Vectors are unit-normalized, so cosine is a dot product
// and a centroid is the normalized mean of its members.
//
// Initialization is farthest-point rather than random: RAPTOR trees are
// expensive to build (one LLM summary per cluster), so a rebuild that silently
// produced a different tree — and different citations — would be worse than a
// marginally worse split. Deterministic in, deterministic out.
func kmeans(vecs [][]float64, k, iters int) [][]int {
	n := len(vecs)
	switch {
	case n == 0 || k <= 0:
		return nil
	case k == 1:
		all := make([]int, n)
		for i := range all {
			all[i] = i
		}
		return [][]int{all}
	case k >= n:
		// Each vector its own cluster.
		out := make([][]int, n)
		for i := range out {
			out[i] = []int{i}
		}
		return out
	}

	centroids := initCentroids(vecs, k)
	assign := make([]int, n)
	for range iters {
		changed := false
		for i, v := range vecs {
			best, bestScore := 0, math.Inf(-1)
			for c, cent := range centroids {
				if s := dot(v, cent); s > bestScore {
					best, bestScore = c, s
				}
			}
			if assign[i] != best {
				assign[i] = best
				changed = true
			}
		}
		if !changed {
			break
		}
		centroids = recenter(vecs, assign, k, centroids)
	}

	clusters := make([][]int, k)
	for i, c := range assign {
		clusters[c] = append(clusters[c], i)
	}
	// Drop empties so callers never summarize nothing.
	out := clusters[:0]
	for _, c := range clusters {
		if len(c) > 0 {
			out = append(out, c)
		}
	}
	return out
}

// initCentroids picks the first vector, then repeatedly the vector least similar
// to anything already chosen. Deterministic and spreads seeds apart.
func initCentroids(vecs [][]float64, k int) [][]float64 {
	centroids := [][]float64{vecs[0]}
	for len(centroids) < k {
		worst, worstScore := -1, math.Inf(1)
		for i, v := range vecs {
			// Similarity to the nearest existing centroid.
			near := math.Inf(-1)
			for _, c := range centroids {
				if s := dot(v, c); s > near {
					near = s
				}
			}
			if near < worstScore {
				worst, worstScore = i, near
			}
		}
		if worst < 0 {
			break
		}
		centroids = append(centroids, vecs[worst])
	}
	return centroids
}

// recenter recomputes each centroid as the normalized mean of its members,
// keeping the previous centroid for clusters that lost every member.
func recenter(vecs [][]float64, assign []int, k int, prev [][]float64) [][]float64 {
	dims := len(vecs[0])
	sums := make([][]float64, k)
	counts := make([]int, k)
	for i := range sums {
		sums[i] = make([]float64, dims)
	}
	for i, v := range vecs {
		c := assign[i]
		counts[c]++
		for d := range v {
			sums[c][d] += v[d]
		}
	}
	out := make([][]float64, k)
	for c := range out {
		if counts[c] == 0 {
			out[c] = prev[c]
			continue
		}
		out[c] = normalize(sums[c])
	}
	return out
}

func dot(a, b []float64) float64 {
	n := min(len(a), len(b))
	var s float64
	for i := range n {
		s += a[i] * b[i]
	}
	return s
}

func normalize(v []float64) []float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return v
	}
	norm := math.Sqrt(sum)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}
