using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public class Shape
    {
  

        // the shape will be stored in data_stack_
        // when dimension is smaller than kStackCache
        // when it is bigger, it will be stored in data_heap_;

        /// <summary>
        ///  size of in stack space 
        /// </summary>
        const int kStackCache = 4;

        /// <summary>
        /// number of dimnsion of the shape
        /// </summary>
        int ndim_;

        /// <summary>
        /// number of cells allocated in data_heap_
        /// </summary>
        int num_heap_allocated_;

        /// <summary>
        /// in stack space used to store shape when it is small
        /// </summary>
        int[] data_stack_ = new int[kStackCache];
 
        /// <summary>
        /// space to store shape when dimension is big
        /// </summary>
        int[] data_heap_;



        /// <summary>
        /// constructor
        /// </summary>

        public Shape()

        {
            ndim_ = 0;
        }

        /// <summary>
        /// constructor from a vector of index
        /// </summary>
        /// <param name="v">the vector</param>
        public Shape(ICollection<int> v)
        {
            ndim_ = v.Count;
            if (ndim_ <= kStackCache)
            {
                data_heap_ = null;
                num_heap_allocated_ = 0;
                Array.Copy(v.ToArray(), data_stack_, v.Count);
            }
            else
            {
                data_heap_ = new int[ndim_];
                num_heap_allocated_ = ndim_;
                Array.Copy(v.ToArray(), data_heap_, v.Count); ;
            }
        }

        /// <summary>
        /// constructor one dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        public Shape(int s1)
        {
            ndim_ = 1;
            if (ndim_ <= kStackCache)
            {
                data_heap_ = null;
                num_heap_allocated_ = 0;
                data_stack_[0] = s1;
            }
            else
            {
                data_heap_ = new int[ndim_];
                num_heap_allocated_ = ndim_;
                data_heap_[0] = s1;
            }
        }

        /// <summary>
        /// constructor two dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        public Shape(int s1, int s2)

        {
            ndim_ = 2;
            if (ndim_ <= kStackCache)
            {
                data_heap_ = null;
                num_heap_allocated_ = 0;
                data_stack_[0] = s1;
                data_stack_[1] = s2;
            }
            else
            {
                data_heap_ = new int[ndim_];
                num_heap_allocated_ = ndim_;
                data_heap_[0] = s1;
                data_heap_[1] = s2;
            }
        }
        /// <summary>
        /// constructor three dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        /// <param name="s3">size of the third dimmension</param>
        public Shape(int s1, int s2, int s3)
        {
            ndim_ = 3;
            if (ndim_ <= kStackCache)
            {
                data_heap_ = null;
                num_heap_allocated_ = 0;
                data_stack_[0] = s1;
                data_stack_[1] = s2;
                data_stack_[2] = s3;
            }
            else
            {
                data_heap_ = new int[ndim_];
                num_heap_allocated_ = ndim_;
                data_heap_[0] = s1;
                data_heap_[1] = s2;
                data_heap_[2] = s3;
            }
        }
  
        /// <summary>
        /// constructor four dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        /// <param name="s3">size of the third dimmension</param>
        /// <param name="s4">size of the fourth dimmension</param>
        public Shape(int s1, int s2, int s3, int s4)
        {
            ndim_ = 4;
            if (ndim_ <= kStackCache)
            {
                data_heap_ = null;
                num_heap_allocated_ = 0;
                data_stack_[0] = s1;
                data_stack_[1] = s2;
                data_stack_[2] = s3;
                data_stack_[3] = s4;
            }
            else
            {
                data_heap_ = new int[ndim_];
                num_heap_allocated_ = ndim_;
                data_heap_[0] = s1;
                data_heap_[1] = s2;
                data_heap_[2] = s3;
                data_heap_[3] = s4;
            }
        }

        /// <summary>
        /// constructor five dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        /// <param name="s3">size of the third dimmension</param>
        /// <param name="s4">size of the fourth dimmension</param>
        /// <param name="s5">size of the fifth dimmension</param>
        public Shape(int s1, int s2, int s3, int s4, int s5)
        {
            ndim_ = 5;
            if (ndim_ <= kStackCache)
            {
                data_heap_ = null;
                num_heap_allocated_ = 0;
                data_stack_[0] = s1;
                data_stack_[1] = s2;
                data_stack_[2] = s3;
                data_stack_[3] = s4;
                data_stack_[4] = s5;
            }
            else
            {
                data_heap_ = new int[ndim_];
                num_heap_allocated_ = ndim_;
                data_heap_[0] = s1;
                data_heap_[1] = s2;
                data_heap_[2] = s3;
                data_heap_[3] = s4;
                data_heap_[5] = s5;
            }
        }
        /// <summary>
        /// constructor from Shape
        /// </summary>
        /// <param name="s">the source shape</param>
        public Shape(Shape s)
        {
            ndim_ = s.ndim_;
            if (ndim_ <= kStackCache)
            {
                data_heap_ = null;
                num_heap_allocated_ = 0;
                Array.Copy(s.data_stack_, data_stack_, ndim_);
            }
            else
            {
                data_heap_ = new int[ndim_];
                num_heap_allocated_ = ndim_;
                Array.Copy(s.data_heap_, data_heap_, ndim_);
            }
        }


  
        /// <summary>
        /// the data content of the shape
        /// </summary>
        /// <returns></returns>
        public ReadOnlyCollection<int> data()
        {
            return Array.AsReadOnly( ndim_ <= kStackCache ? data_stack_ : data_heap_);
        }

        /// <summary>
        ///  return number of dimension of the tensor inside
        /// </summary>
        /// <returns></returns>
        int ndim()
        {
            return ndim_;
        }

        /// <summary>
        /// get corresponding index
        /// </summary>
        /// <param name="i">dimension index</param>
        /// <returns>the corresponding dimension size</returns>
        int this[int i]
        {
            get { return data()[i]; }
        }


        /// <summary>
        /// total number of elements in the tensor
        /// </summary>
        /// <returns></returns>
        int Size()
        {
            int size = 1;
            ReadOnlyCollection<int> d = this.data();
            for (int i = 0; i < ndim_; ++i)
            {
                size *= d[i];
            }
            return size;
        }

        /// <summary>
        /// whether two shape equals
        /// </summary>
        /// <param name="l">the shape to compare against</param>
        /// <param name="r">the shape to compare against</param>
        /// <returns></returns>
        public static bool operator ==(Shape l, Shape r)
        {
            if (l == null) return false;
            if (r == null) return false;
            if (l.ndim_ != r.ndim_) return false;
            if (l.ndim_ <= kStackCache)
            {
                for (int i = 0; i < l.ndim_; ++i)
                {
                    if (l.data_stack_[i] != r.data_stack_[i]) return false;
                }
            }
            else
            {
                for (int i = 0; i < l.ndim_; ++i)
                {
                    if (l.data_heap_[i] != r.data_heap_[i]) return false;
                }
            }
            return true;
        }

        /// <summary>
        /// whether two shape not equals
        /// </summary>
        /// <param name="l"></param>
        /// <param name="r"></param>
        /// <returns></returns>
        public static bool operator !=(Shape l, Shape r )
        {
            return !(l == r);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Shape)obj);
        }

        public override int GetHashCode()
        {

            int hash = ndim_.GetHashCode();
            if (ndim_ <= kStackCache)
            {
                for (int i = 0; i < ndim_; ++i)
                {
                    hash ^= data_stack_[i].GetHashCode();
                }
            }
            else
            {
                for (int i = 0; i < ndim_; ++i)
                {
                    hash ^= data_heap_[i].GetHashCode();
                }
            }
            return hash;
        }

        private bool Equals(Shape other)
        {
            return this == other;
        }


        /// <summary>
        /// internal function to set the dimension
        /// </summary>
        /// <param name="dim">dim the dimension of the shape</param>
        void SetDim(int dim)
        {
            if (dim > kStackCache &&
              dim > num_heap_allocated_)
            {
                // data_heap_ can be NULL

                data_heap_ = new int[dim];
                num_heap_allocated_ = dim;
            }
            ndim_ = dim;
        }
    }
}

