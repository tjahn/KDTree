#pragma once

#include <vector>
#include <numeric>
#include <cassert>

namespace kdtree
{

    template <int _DIM = -1>
    class KDTree
    {
    public:
        using Scalar = float;
        using IndexType = int32_t;
        static constexpr int BLOCK_SIZE = 8; // block size in kd-tree

        class Navigator
        {
        public:
            size_t blocksize;
            const Scalar *pointer;

        public:
            bool go_left(int dim)
            {
                blocksize = blocksize >> 1;
                if (blocksize)
                    pointer += dim;
                return blocksize;
            }

            bool go_right(int dim)
            {
                auto tmp = blocksize >> 1; // blocksize of lhs
                blocksize -= tmp;          // blocksize of rhs + 1
                if (blocksize)             // if has rhs block
                {
                    blocksize--; // blocksize of rhs
                    pointer += dim * (tmp + 1);
                }
                return blocksize;
            }

            Navigator get_left(int dim) const
            {
                auto res = *this;
                res.go_left(dim);
                return res;
            }

            Navigator get_right(int dim) const
            {
                auto res = *this;
                res.go_right(dim);
                return res;
            }
        };

    public:
        KDTree(int dim = _DIM) : dim(_DIM <= 0 ? dim : _DIM){};

        template <typename Iterator>
        void build(const Iterator begin, const Iterator end, const bool multithreaded = false);

        inline int get_dim() const
        {
            return dim;
        }

        inline size_t get_size() const
        {
            return data.size() / dim;
        }

        Navigator get_navigator() const
        {
            return Navigator{idxs.size(), data.data()};
        }

        const std::vector<Scalar> &getInternalData() const
        {
            return data;
        }

    private:
        void _build(IndexType *begin, IndexType *end, int depth);
        void _build_multithreaded(IndexType *begin, IndexType *end, int depth);

        template <typename Iterator>
        Iterator center(const Iterator begin, const Iterator end)
        {
            size_t halfSize = (end - begin) / 2;
            auto median = begin + halfSize;
            return median;
        }

    private:
        const int dim;

        std::vector<IndexType> idxs;
        std::vector<Scalar> data;
    };

    template <int DIM>
    template <typename Iterator>
    void KDTree<DIM>::build(const Iterator begin, const Iterator end, const bool multithreaded)
    {
        assert((end - begin) % get_dim() == 0);
        idxs.resize((end - begin) / get_dim());
        std::iota(idxs.begin(), idxs.end(), 0);
        data.clear();
        data.insert(data.begin(), begin, end);

        if (multithreaded)
        {
#pragma omp parallel
#pragma omp single nowait
            {
                _build_multithreaded(idxs.data(), idxs.data() + idxs.size(), 0);
            }
        }
        else
        {
            _build(idxs.data(), idxs.data() + idxs.size(), 0);
        }

        // reorder data inplace
        std::vector<Scalar> sortedData(data.size());
        for (int idx = 0; idx < idxs.size(); ++idx)
        {
            auto srcIdx = idxs[idx];
            for (int j = 0; j < dim; ++j)
                sortedData[idx * dim + j] = data[srcIdx * dim + j];
        }
        data = sortedData;
        std::iota(idxs.begin(), idxs.end(), 0);
    }

    template <int DIM>
    void KDTree<DIM>::_build(IndexType *begin, IndexType *end, int depth)
    {
        if (end - begin < 2)
        { // nothing to sort anymore
            return;
        }

        const auto axis = depth % dim;
        auto median = center(begin, end);

        std::nth_element(begin, median, end, [axis, this](const IndexType small, const IndexType large) -> bool
                         { return data[small * dim + axis] < data[large * dim + axis]; });
        std::swap(*begin, *median);

        _build(begin + 1, median + 1, depth + 1);
        _build(median + 1, end, depth + 1);
    }

    template <int DIM>
    void KDTree<DIM>::_build_multithreaded(IndexType *begin, IndexType *end, int depth)
    {

        if (end - begin < 512 || depth >= 5)
        { // fall back to single threading when reaching certain cell-size
            return _build(begin, end, depth);
        }

        const auto axis = depth % dim;
        auto median = center(begin, end);

        std::nth_element(begin, median, end, [axis, this](const IndexType small, const IndexType &large) -> bool
                         { return data[small * dim + axis] < data[large * dim + axis]; });
        std::swap(*begin, *median);

        {
#pragma omp task
            _build_multithreaded(begin + 1, median + 1, depth + 1);
#pragma omp task
            _build_multithreaded(median + 1, end, depth + 1);
#pragma omp taskwait
        }
    }

} // namespace kdtree