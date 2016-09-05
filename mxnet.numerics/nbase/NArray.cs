using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{

    public class NArray<T, TC, TOut>
        where T : new()
        where TC : ICalculator<T>, new()
        where TOut : NArray<T, TC, TOut> , new()
    {
        private static readonly TC Calculator = new TC();

        private Shape _storage_shape;
        public Shape shape { get; protected set; }

        protected T[] storage;
        private Slice[] _slice;

        public T[] data => storage;

        public GCHandle GetDataGcHandle()
        {
          return  GCHandle.Alloc(storage, GCHandleType.Pinned);
        }

        public NArray()
        {
            
        }

        public NArray(Shape shape)
        {
            this.shape = new Shape(shape);
            this._storage_shape = this.shape;
            storage = new T[this.shape.size];
            _slice = Slice.FromShape(this.shape.data);
        }
        public NArray(Shape shape, T[] data)
        {
            this.shape = new Shape(shape);
            this._storage_shape = this.shape;
            storage = new T[this.shape.size];
            Array.Copy(data, storage, Math.Min(data.Length, storage.Length));
            _slice = Slice.FromShape(this.shape.data);
        }

        private TOut SilceData(params Slice[] slice)
        {
            var src_dim = _storage_shape.data;
            var tslice = slice;
            var dst_dim = shape.data;

            var ret = new TOut();
            ret.shape = new Shape(dst_dim);
            ret.storage = new T[ret.shape.size];
            ArrayCopy(storage, ret.storage, 0, storage.Length, 0, ret.storage.Length,
                tslice, src_dim,
                dst_dim);
            return ret;
        }

        static void ArrayCopy(T[] src, T[] dst, int src_start, int src_end, int dst_start, int dst_end,
            Slice[] slice, uint[] src_dim,uint[] dst_dim
            )
        {
            var first_slice = slice.FirstOrDefault();
            if (first_slice != null)
            {
                int src_pad = CalcPad(src_dim.Skip(1).ToArray());
                int dst_index = 0;
                var dst_pad = CalcPad(dst_dim.Skip(1).ToArray());
                for (var i = first_slice.start;
        
                    ((first_slice.step > 0) ? (i < first_slice.end) : (i > first_slice.end));
                    i += first_slice.step)
                {
                    ArrayCopy(src, dst,
                        src_start + i * src_pad,
                        src_start + (i + 1) * src_pad,
                        dst_start + dst_index * dst_pad,
                        dst_start + (dst_index + 1) * dst_pad,
                        slice.Skip(1).ToArray(),
                        src_dim.Skip(1).ToArray(),
                        dst_dim.Skip(1).ToArray());
                    dst_index++;
                }
            }
            else
            {
                Array.Copy(src, src_start, dst, dst_start, dst_end - dst_start);
            }
        }


        private IEnumerable<T> SilceDataYield(params Slice[] slice)
        {
            var src_dim = _storage_shape.data;
            var tslice = slice;
            var dst_dim = shape.data;


            var dst_shape = new Shape(dst_dim);

            var ret1 = ArrayYield(storage, 0, 0, (int)dst_shape.size,
                 tslice, src_dim, dst_dim);
            return ret1;
        }


        static IEnumerable<T> ArrayYield(T[] src, int src_start, int dst_start, int dst_end,Slice[] slice, uint[] src_dim, uint[] dst_dim )
        {
            var first_slice = slice.FirstOrDefault();
            if (first_slice != null)
            {
                int src_pad = CalcPad(src_dim.Skip(1).ToArray());
                int dst_index = 0;
                var dst_pad = CalcPad(dst_dim.Skip(1).ToArray());
                for (var i = first_slice.start;

                    ((first_slice.step > 0) ? (i < first_slice.end) : (i > first_slice.end));
                    i += first_slice.step)
                {
                    var ay = ArrayYield(src,
                       src_start + i * src_pad,
                       dst_start + dst_index * dst_pad,
                       dst_start + (dst_index + 1) * dst_pad,
                       slice.Skip(1).ToArray(),
                       src_dim.Skip(1).ToArray(),
                       dst_dim.Skip(1).ToArray());

                    foreach (var item in ay)
                    {
                        yield return item;
                    }

                    dst_index++;
                }
            }
            else
            {
                var count = src_start + (dst_end - dst_start);
                for (int i = src_start; i < count; i++)
                {
                    yield return src[i];
                }
        
            }
        }


        private static int CalcPad(uint[] src_dim)
        {
            return (int)src_dim.Aggregate((long)1, (l, r) => l * r);
        }

        public virtual TOut this[params Slice[] slice]
        {
            get
            {
               // return SilceData(slice);

                var src_dim = shape.data;
                var tslice = slice.Select((x, i) => x.Translate(src_dim[i])).ToArray();
                var dst_dim_temp = tslice.Select(s => (uint)s.size).ToArray();
                var dst_dim = (uint[])src_dim.Clone();
                Array.Copy(dst_dim_temp, dst_dim, dst_dim_temp.Length);

                var ret = new TOut();
                ret._storage_shape = this._storage_shape;
                ret.shape = new Shape(dst_dim);
                ret.storage = storage;

                ret._slice = _slice.Zip(tslice, (l, r) => l.SubSlice(r))
                    .Concat(_slice.Skip(tslice.Count())).ToArray();
                return ret;
            }
        }

   

        public TOut this[int d0]
        {
            get
            {
                var ret = new TOut();
                ret.shape = new Shape(shape.data.Skip(1).ToArray());
                ret.storage = new T[ret.shape.size];

                int retindex = 0;
                int d1 =(int) shape.data[1];
                for (int i = 0; i < d1; i++)
                {
                    ret.storage[retindex] = storage[d0 * d1 + i];
                    retindex++;
                }
                return ret;
            }
        }

        public TOut Flat()
        {
            var ret = new TOut();
            ret.shape = new Shape(shape.size);
            var temp = SilceData(_slice);
            ret.storage = temp.storage;
            return ret;
        }

        public TOut Flat2()
        {
            var ret = new TOut();
            ret.shape = new Shape(shape.size);
            ret.storage = SilceDataYield(_slice).ToArray();
            return ret;
        }

        public TOut Compare(NArray<T, TC, TOut> other)
        {
            if (shape != other.shape)
            {
                TOut retfalse = new TOut()
                {
                    shape = new Shape(1),
                    storage = new T[] { (T)Convert.ChangeType(0, typeof(T)) }
                };
                return retfalse;
            }


            TOut ret = new TOut
            {
                shape = shape,
                storage = this.storage
                    .Select((x, i) => Calculator.Compare(x, other.storage[i])).ToArray()
  
            };
            return ret;
        }
        #region

        public T Sum()
        {
            return Calculator.Sum(Flat().storage);
        }


        public int Argmax()
        {
            return Calculator.Argmax(Flat().storage);
        }



        #endregion
    }
}
