using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{

    public partial class NArray<T, TC, TOut>
        where T : new()
        where TC : ICalculator<T>, new()
        where TOut : NArray<T, TC, TOut> , new()
    {
        private static readonly TC Calculator = new TC();

        private Shape _storage_shape;
        public Shape shape { get; protected set; }

        protected T[] storage;
        private Slice[] _slice;

        public T[] data => SilceGetData(_slice);

        public GCHandle GetDataGcHandle()
        {
          return  GCHandle.Alloc(storage, GCHandleType.Pinned);
        }

        public NArray()
        {
            
        }

        public NArray(Shape shape)
        {
            Init(shape);
        }
        public NArray(Shape shape, T[] data)
        {
            Init(shape);
            Array.Copy(data, storage, Math.Min(data.Length, storage.Length));
        }
        /// <summary>
        /// init with zero cpoy
        /// </summary>
        /// <param name="shape_input"></param>
        /// <param name="data_input"></param>

        private void Init(Shape shape_input, T[] data_input =null)
        {
            this.shape = new Shape(shape_input);
            this._storage_shape = this.shape;
            storage = data_input ?? new T[this.shape.size];
            _slice = Slice.FromShape(this.shape.data);
        }


        private TOut SliceToTout(Slice[] slice)
        {
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




        private void SilceSetData(TOut value, params Slice[] slice)
        {
            if (shape != value.shape)
            {
                throw new ArgumentException("shape not match");
            }
            var src_dim = _storage_shape.data;
            var dst_dim = value.shape.data;

            ArrayCopy(storage, value.data, 0, storage.Length, 0, 0,
                slice, src_dim, dst_dim, true);

        }

        private T[] SilceGetData(params Slice[] slice)
        {
            var src_dim = _storage_shape.data;
            var tslice = slice;
            var dst_dim = shape.data;


            var ret_vshape = new Shape(dst_dim);
            var local_storage = new T[ret_vshape.size];
            ArrayCopy(storage, local_storage, 0, storage.Length, 0, local_storage.Length,
                tslice, src_dim,
                dst_dim);
            return local_storage;
        }

        static void ArrayCopy(T[] src, T[] dst, 
            int src_start, int src_end, 
            int dst_start, int dst_end,
            Slice[] slice, uint[] src_dim,
            uint[] dst_dim ,bool copytosrc =false
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
                        dst_dim.Skip(1).ToArray(),
                        copytosrc);
                    dst_index++;
                }
            }
            else
            {
                if (!copytosrc)
                {
                    Array.Copy(src, src_start, dst, dst_start, dst_end - dst_start);
                }
                else
                {
                    Array.Copy(dst, dst_start, src, src_start,  dst_end - dst_start);
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
                var ret = SliceToTout(slice);
                return ret;
            }
            set
            {
                var ret = SliceToTout(slice);

                ret.Assign(value);
            }
        }

   

        public TOut this[params int[][] indexs]
        {
            get
            {
                if (indexs.Any(o => o.Length != indexs.First().Length))
                {
                    throw new ArgumentException("");
                }


                var getshape = new Shape(new uint[] { 1} .Concat(shape.data.Skip(indexs.Length)).ToArray());


                var ret = new TOut();
                ret.Init(new Shape(
                    new uint[]{(uint) indexs[0].Length }  
                    .Concat(shape.data.Skip(indexs.Length)).ToArray()
                    ));
                var count = indexs.First().Length;

                for (int i = 0; i < count; i++)
                {
                    var slices = indexs.Select(s => (Slice) s[i]).ToArray();

                    ret[(Slice)i] = this[slices].ReShape(getshape);
                }
                return ret;
            }
        }

        public T this[params int[] indexs]
        {
            get
            {
                var retindex = GetIndexWithScalar(indexs);
                return storage[retindex];
            }
            set
            {
                var retindex = GetIndexWithScalar(indexs);
                storage[retindex] = value;
            }
        }



        private int GetIndexWithScalar(int[] indexs)
        {
            int retindex = 0;
            var src_dim = shape.data;
            for (int i = 0; i < indexs.Length; i++)
            {
                var index = indexs[i];
                var s = _slice[i];
                int src_pad = CalcPad(src_dim.Skip(i + 1).ToArray());

                index = s.step > 0 ? s.start + index : s.start - index;
                retindex += (index*src_pad);
            }
            return retindex;
        }

        private void Assign(TOut value)
        {
            SilceSetData(value, _slice);
        }


        public TOut Flat()
        {
            var ret = new TOut();
            ret.Init(new Shape(shape.size), data);
            return ret;
        }

        public TOut Compare(NArray<T, TC, TOut> other)
        {
            if (shape != other.shape)
            {
                TOut retfalse = new TOut();
                retfalse.Init(new Shape(1), new T[] {(T) Convert.ChangeType(0, typeof(T))});
                return retfalse;
            }


            TOut ret = new TOut();
            ret.Init(shape, this.storage
                .Select((x, i) => Calculator.Compare(x, other.storage[i])).ToArray());
            return ret;
        }


        public TOut ReShape(Shape shape_input)
        {
            if (shape_input.size != shape.size)
            {
                throw new ArgumentException("size mistake");
            }
            TOut ret = new TOut();
            ret.Init(shape_input, data);
            return ret;
        }

        #region

        public T Sum()
        {
            return Calculator.Sum(data);
        }

        private static void Argmax(NArray<T, TC, TOut> src,TOut dst, uint[] src_dim, int dimindex, int axis ,  List<Slice> src_slice, List<int> dst_index)
        {
            if (dimindex < src_dim.Length)
            {
                src_slice.Add(0);
                if (axis == dimindex)
                {
                    src_slice[dimindex] = ":";
                    Argmax(src, dst, src_dim, dimindex + 1, axis, src_slice, dst_index);
                }
                else
                {
                    dst_index.Add(0);
                    var dst_index_last = dst_index.Count - 1;
                    var current_dim = src_dim[dimindex];

                    for (int i = 0; i < current_dim; i++)
                    {
                        src_slice[dimindex] = i;
                        dst_index[dst_index_last] = i;
                        Argmax(src, dst, src_dim, dimindex + 1, axis, src_slice, dst_index);
                    }
                    dst_index.RemoveAt(dst_index.Count - 1);
                }
                src_slice.RemoveAt(src_slice.Count - 1);
            }
            else
            {
                dst[dst_index.ToArray()] =(T)Convert.ChangeType( src[src_slice.ToArray()].Argmax(),typeof(T));
            }

        }

        public int Argmax()
        {

            return Calculator.Argmax(data);

        }

        public TOut Argmax(int axis )
        {
            var src_dim = shape.data.ToList();
            src_dim.RemoveAt(axis);
            TOut ret = new TOut();
            ret.Init(new Shape(src_dim.ToArray()));
            List<Slice> src_slice = new List<Slice>();
            List<int> dst_index = new List<int>();
            Argmax(this, ret, shape.data, 0, axis, src_slice, dst_index);
            return ret;
        }

        public TOut Log()
        {
            TOut ret = new TOut();
            ret.Init(shape, Calculator.Log(data));
            return ret;
        }

        public T Mean()
        {
            return Calculator.Mean(data);
        }

        #endregion
    }
}
