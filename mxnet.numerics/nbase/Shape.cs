using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public class Shape
    {


        /// <summary>
        /// number of dimnsion of the shape
        /// </summary>
        readonly uint _ndim;

        /// <summary>
        /// space to store shape when dimension is big
        /// </summary>
        readonly uint[] _dataHeap;


        /// <summary>
        /// constructor
        /// </summary>
        public Shape()

        {
            _ndim = 0;
        }

        /// <summary>
        /// constructor from a vector of index
        /// </summary>
        /// <param name="v">the vector</param>
        public Shape(ICollection<uint> v)
        {
            _ndim = (uint)v.Count;
            _dataHeap = new uint[_ndim];
            Array.Copy(v.ToArray(), _dataHeap, v.Count); ;
        }

        /// <summary>
        /// constructor one dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        public Shape(uint s1)
        {
            _ndim = 1;
            _dataHeap = new uint[_ndim];
            _dataHeap[0] = s1;
        }

        /// <summary>
        /// constructor two dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        public Shape(uint s1, uint s2)

        {
            _ndim = 2;
            _dataHeap = new uint[_ndim];
            _dataHeap[0] = s1;
            _dataHeap[1] = s2;

        }
        /// <summary>
        /// constructor three dimmension shape
        /// </summary>
        /// <param name="s1">size of the first dimmension</param>
        /// <param name="s2">size of the second dimmension</param>
        /// <param name="s3">size of the third dimmension</param>
        public Shape(uint s1, uint s2, uint s3)
        {
            _ndim = 3;
            _dataHeap = new uint[_ndim];
            _dataHeap[0] = s1;
            _dataHeap[1] = s2;
            _dataHeap[2] = s3;
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
            _ndim = 4;
            _dataHeap = new uint[_ndim];
            _dataHeap[0] = s1;
            _dataHeap[1] = s2;
            _dataHeap[2] = s3;
            _dataHeap[3] = s4;
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
            _ndim = 5;
            _dataHeap = new uint[_ndim];
            _dataHeap[0] = s1;
            _dataHeap[1] = s2;
            _dataHeap[2] = s3;
            _dataHeap[3] = s4;
            _dataHeap[5] = s5;
        }
        /// <summary>
        /// constructor from Shape
        /// </summary>
        /// <param name="s">the source shape</param>
        public Shape(Shape s)
        {
            _ndim = s._ndim;
            _dataHeap = new uint[_ndim];
            Array.Copy(s._dataHeap, _dataHeap, _ndim);
        }



        /// <summary>
        /// the data content of the shape
        /// </summary>
        /// <returns></returns>
        public uint[] Data => _dataHeap;

        /// <summary>
        ///  return number of dimension of the tensor inside
        /// </summary>
        /// <returns></returns>
        public uint Ndim => _ndim;

        /// <summary>
        /// get corresponding index
        /// </summary>
        /// <param name="i">dimension index</param>
        /// <returns>the corresponding dimension size</returns>
        public uint this[int i] => _dataHeap[i];


        /// <summary>
        /// total number of elements in the tensor
        /// </summary>
        /// <returns></returns>
        public uint Size
        {
            get
            {
                uint size = 1;
                var d = this.Data;
                for (int i = 0; i < _ndim; ++i)
                {
                    size *= d[i];
                }
                return size;
            }
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
            return l.Equals(r);
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
            int hash = _ndim.GetHashCode();
            for (int i = 0; i < _ndim; ++i)
            {
                hash ^= _dataHeap[i].GetHashCode();
            }
            return hash;
        }

        private bool Equals(Shape other)
        {
            var l = this;
            var r = other;
            if (ReferenceEquals(l, r)) return true;
            if (ReferenceEquals(r, null)) return false;

            if (l._ndim != r._ndim) return false;
            for (int i = 0; i < l._ndim; ++i)
            {
                if (l._dataHeap[i] != r._dataHeap[i]) return false;
            }
            return true;
        }


        public override string ToString()
        {
            string ret = "";
            ret += "(";
            for (int i = 0; i < Ndim; ++i)
            {
                if (i != 0) ret += ",";
                ret += this[i];
            }
            // python style tuple
            if (Ndim == 1) ret += ",";
            ret += ")";
            return ret;
        }


        public static implicit operator uint[] (Shape obj)
        {
            return obj.Data;
        }
    }
}

