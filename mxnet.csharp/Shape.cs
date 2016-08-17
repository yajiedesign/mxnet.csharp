using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    class Shape
    {
 



        // the shape will be stored in data_stack_
        // when dimension is smaller than kStackCache
        // when it is bigger, it will be stored in data_heap_;
        /*! \brief size of in stack space */
        const int kStackCache = 4;
        /*! \brief number of dimnsion of the shape */
        int ndim_;
        /*! \brief number of cells allocated in data_heap_ */
        int num_heap_allocated_;
        /*! \brief in stack space used to store shape when it is small */
        int[] data_stack_ = new int[kStackCache];
        /*! \brief space to store shape when dimension is big*/
        int[] data_heap_;


        /*! \brief constructor */

   public     Shape()

        {
            ndim_ = 0;
        }
        /*!
        * \brief constructor from a vector of index_t
        * \param v the vector
        */
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
        /*!
        * \brief constructor one dimmension shape
        * \param s1 size of the first dimmension
        */
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
        /*!
        * \brief constructor two dimmension shape
        * \param s1 size of the first dimmension
        * \param s2 size of the second dimmension
        */
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
        /*!
        * \brief constructor three dimmension shape
        * \param s1 size of the first dimmension
        * \param s2 size of the second dimmension
        * \param s3 size of the third dimmension
        */
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
        /*!
        * \brief constructor four dimmension shape
        * \param s1 size of the first dimmension
        * \param s2 size of the second dimmension
        * \param s3 size of the third dimmension
        * \param s4 size of the fourth dimmension
        */
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
        /*!
        * \brief constructor five dimmension shape
        * \param s1 size of the first dimmension
        * \param s2 size of the second dimmension
        * \param s3 size of the third dimmension
        * \param s4 size of the fourth dimmension
        * \param s5 size of the fifth dimmension
        */
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
        /*!
        * \brief constructor from Shape
        * \param s the source shape
        */
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
 

        /*! \return the data content of the shape */
        public ReadOnlyCollection<int> data()
        {
            return Array.AsReadOnly( ndim_ <= kStackCache ? data_stack_ : data_heap_);
        }
        /*! \brief return number of dimension of the tensor inside */
        int ndim()
        {
            return ndim_;
        }
        /*!
        * \brief get corresponding index
        * \param i dimension index
        * \return the corresponding dimension size
        */
        int this[int i]
        {
            get { return data()[i]; }
        }

        /*! \brief total number of elements in the tensor */
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
        /*!
        * \return whether two shape equals
        * \param s the shape to compare against
        */
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
        /*!
        * \return whether two shape not equals
        * \param s the shape to compare against
        */
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


        /*!
        * \brief internal function to set the dimension
        * \param dim the dimension of the shape
        */
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

