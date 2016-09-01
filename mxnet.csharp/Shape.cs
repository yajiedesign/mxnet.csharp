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
        const int KStackCache = 4;

        /// <summary>
        /// number of dimnsion of the shape
        /// </summary>
        uint _ndim_;

        /// <summary>
        /// number of cells allocated in data_heap_
        /// </summary>
        uint _num_heap_allocated_;

        /// <summary>
        /// in stack space used to store shape when it is small
        /// </summary>
        readonly uint[] _data_stack_ = new uint[KStackCache];

        /// <summary>
        /// space to store shape when dimension is big
        /// </summary>
        uint[] _data_heap_;



        /// <summary>
        /// constructor
        /// </summary>

        public Shape()

        {
            _ndim_ = 0;
        }

        /// <summary>
        /// constructor from a vector of index
        /// </summary>
        /// <param name="v">the vector</param>
        public Shape(ICollection<uint> v)
        {
            _ndim_ = (uint)v.Count;
            if (_ndim_ <= KStackCache)
            {
                _data_heap_ = null;
                _num_heap_allocated_ = 0;
                Array.Copy(v.ToArray(), _data_stack_, v.Count);
            }
            else
            {
                _data_heap_ = new uint[_ndim_];
                _num_heap_allocated_ = _ndim_;
                Array.Copy(v.ToArray(), _data_heap_, v.Count); ;
            }
        }

        /// <summary>
        /// constructor one dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        public Shape(uint s1)
        {
            _ndim_ = 1;
            if (_ndim_ <= KStackCache)
            {
                _data_heap_ = null;
                _num_heap_allocated_ = 0;
                _data_stack_[0] = s1;
            }
            else
            {
                _data_heap_ = new uint[_ndim_];
                _num_heap_allocated_ = _ndim_;
                _data_heap_[0] = s1;
            }
        }

        /// <summary>
        /// constructor two dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        public Shape(uint s1, uint s2)

        {
            _ndim_ = 2;
            if (_ndim_ <= KStackCache)
            {
                _data_heap_ = null;
                _num_heap_allocated_ = 0;
                _data_stack_[0] = s1;
                _data_stack_[1] = s2;
            }
            else
            {
                _data_heap_ = new uint[_ndim_];
                _num_heap_allocated_ = _ndim_;
                _data_heap_[0] = s1;
                _data_heap_[1] = s2;
            }
        }
        /// <summary>
        /// constructor three dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        /// <param name="s3">size of the third dimmension</param>
        public Shape(uint s1, uint s2, uint s3)
        {
            _ndim_ = 3;
            if (_ndim_ <= KStackCache)
            {
                _data_heap_ = null;
                _num_heap_allocated_ = 0;
                _data_stack_[0] = s1;
                _data_stack_[1] = s2;
                _data_stack_[2] = s3;
            }
            else
            {
                _data_heap_ = new uint[_ndim_];
                _num_heap_allocated_ = _ndim_;
                _data_heap_[0] = s1;
                _data_heap_[1] = s2;
                _data_heap_[2] = s3;
            }
        }

        /// <summary>
        /// constructor four dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        /// <param name="s3">size of the third dimmension</param>
        /// <param name="s4">size of the fourth dimmension</param>
        public Shape(uint s1, uint s2, uint s3, uint s4)
        {
            _ndim_ = 4;
            if (_ndim_ <= KStackCache)
            {
                _data_heap_ = null;
                _num_heap_allocated_ = 0;
                _data_stack_[0] = s1;
                _data_stack_[1] = s2;
                _data_stack_[2] = s3;
                _data_stack_[3] = s4;
            }
            else
            {
                _data_heap_ = new uint[_ndim_];
                _num_heap_allocated_ = _ndim_;
                _data_heap_[0] = s1;
                _data_heap_[1] = s2;
                _data_heap_[2] = s3;
                _data_heap_[3] = s4;
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
        public Shape(uint s1, uint s2, uint s3, uint s4, uint s5)
        {
            _ndim_ = 5;
            if (_ndim_ <= KStackCache)
            {
                _data_heap_ = null;
                _num_heap_allocated_ = 0;
                _data_stack_[0] = s1;
                _data_stack_[1] = s2;
                _data_stack_[2] = s3;
                _data_stack_[3] = s4;
                _data_stack_[4] = s5;
            }
            else
            {
                _data_heap_ = new uint[_ndim_];
                _num_heap_allocated_ = _ndim_;
                _data_heap_[0] = s1;
                _data_heap_[1] = s2;
                _data_heap_[2] = s3;
                _data_heap_[3] = s4;
                _data_heap_[5] = s5;
            }
        }
        /// <summary>
        /// constructor from Shape
        /// </summary>
        /// <param name="s">the source shape</param>
        public Shape(Shape s)
        {
            _ndim_ = s._ndim_;
            if (_ndim_ <= KStackCache)
            {
                _data_heap_ = null;
                _num_heap_allocated_ = 0;
                Array.Copy(s._data_stack_, _data_stack_, _ndim_);
            }
            else
            {
                _data_heap_ = new uint[_ndim_];
                _num_heap_allocated_ = _ndim_;
                Array.Copy(s._data_heap_, _data_heap_, _ndim_);
            }
        }



        /// <summary>
        /// the data content of the shape
        /// </summary>
        /// <returns></returns>
        public uint[] Data()
        {
            return ((uint[])(_ndim_ <= KStackCache ? _data_stack_ : _data_heap_)).Take((int)_ndim_).ToArray();
        }

        /// <summary>
        ///  return number of dimension of the tensor inside
        /// </summary>
        /// <returns></returns>
        public uint Ndim()
        {
            return _ndim_;
        }

        /// <summary>
        /// get corresponding index
        /// </summary>
        /// <param name="i">dimension index</param>
        /// <returns>the corresponding dimension size</returns>
        public uint this[int i]
        {
            get { return Data()[i]; }
        }


        /// <summary>
        /// total number of elements in the tensor
        /// </summary>
        /// <returns></returns>
        public uint Size()
        {
            uint size = 1;
            var d = this.Data();
            for (int i = 0; i < _ndim_; ++i)
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
            if (ReferenceEquals(l, r)) return true;
            if (ReferenceEquals(l, null)) return false;
            if (ReferenceEquals(r, null)) return false;

            if (l._ndim_ != r._ndim_) return false;
            if (l._ndim_ <= KStackCache)
            {
                for (int i = 0; i < l._ndim_; ++i)
                {
                    if (l._data_stack_[i] != r._data_stack_[i]) return false;
                }
            }
            else
            {
                for (int i = 0; i < l._ndim_; ++i)
                {
                    if (l._data_heap_[i] != r._data_heap_[i]) return false;
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
        public static bool operator !=(Shape l, Shape r)
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

            int hash = _ndim_.GetHashCode();
            if (_ndim_ <= KStackCache)
            {
                for (int i = 0; i < _ndim_; ++i)
                {
                    hash ^= _data_stack_[i].GetHashCode();
                }
            }
            else
            {
                for (int i = 0; i < _ndim_; ++i)
                {
                    hash ^= _data_heap_[i].GetHashCode();
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
        void SetDim(uint dim)
        {
            if (dim > KStackCache &&
              dim > _num_heap_allocated_)
            {
                // data_heap_ can be NULL

                _data_heap_ = new uint[dim];
                _num_heap_allocated_ = dim;
            }
            _ndim_ = dim;
        }
        public override string ToString()
        {
            string ret = "";
            ret += "(";
            for (int i = 0; i < Ndim(); ++i)
            {
                if (i != 0) ret += ",";
                ret += this[i];
            }
            // python style tuple
            if (Ndim() == 1) ret += ",";
            ret += ")";
            return ret;
        }


        public static implicit operator uint[] (Shape obj)
        {
            return obj.Data();
        }
    }
}

