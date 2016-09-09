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

        private Shape _storageShape;
        public Shape Shape { get; protected set; }

        protected T[] Storage;
        private Slice[] _slice;

        public T[] Data => SilceGetData(_slice);

        public GCHandle GetDataGcHandle()
        {
          return  GCHandle.Alloc(Storage, GCHandleType.Pinned);
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
            Array.Copy(data, Storage, Math.Min(data.Length, Storage.Length));
        }
        /// <summary>
        /// init with zero cpoy
        /// </summary>
        /// <param name="shapeInput"></param>
        /// <param name="dataInput"></param>

        private void Init(Shape shapeInput, T[] dataInput =null)
        {
            this.Shape = new Shape(shapeInput);
            this._storageShape = this.Shape;
            Storage = dataInput ?? new T[this.Shape.Size];
            _slice = Slice.FromShape(this.Shape.Data);
        }


        private TOut SliceToTout(Slice[] slice)
        {
            var srcDim = Shape.Data;
            var tslice = slice.Select((x, i) => x.Translate(srcDim[i])).ToArray();
            var dstDimTemp = tslice.Select(s => (uint)s.Size).ToArray();
            var dstDim = (uint[])srcDim.Clone();
            Array.Copy(dstDimTemp, dstDim, dstDimTemp.Length);

            var ret = new TOut();
            ret._storageShape = this._storageShape;
            ret.Shape = new Shape(dstDim);
            ret.Storage = Storage;

            ret._slice = _slice.Zip(tslice, (l, r) => l.SubSlice(r))
                .Concat(_slice.Skip(tslice.Count())).ToArray();

            return ret;
        }




        private void SilceSetData(TOut value, params Slice[] slice)
        {
            if (Shape != value.Shape)
            {
                throw new ArgumentException("shape not match");
            }
            var srcDim = _storageShape.Data;
            var dstDim = value.Shape.Data;

            ArrayCopy(Storage, value.Data, 0, Storage.Length, 0, 0,
                slice, srcDim, dstDim, true);

        }

        private T[] SilceGetData(params Slice[] slice)
        {
            var srcDim = _storageShape.Data;
            var tslice = slice;
            var dstDim = Shape.Data;


            var retVshape = new Shape(dstDim);
            var localStorage = new T[retVshape.Size];
            ArrayCopy(Storage, localStorage, 0, Storage.Length, 0, localStorage.Length,
                tslice, srcDim,
                dstDim);
            return localStorage;
        }

        static void ArrayCopy(T[] src, T[] dst, 
            int srcStart, int srcEnd, 
            int dstStart, int dstEnd,
            Slice[] slice, uint[] srcDim,
            uint[] dstDim ,bool copytosrc =false
            )
        {
            var firstSlice = slice.FirstOrDefault();
            if (firstSlice != null)
            {
                int srcPad = CalcPad(srcDim.Skip(1).ToArray());
                int dstIndex = 0;
                var dstPad = CalcPad(dstDim.Skip(1).ToArray());
                for (var i = firstSlice.Start;
        
                    ((firstSlice.Step > 0) ? (i < firstSlice.End) : (i > firstSlice.End));
                    i += firstSlice.Step)
                {
                    ArrayCopy(src, dst,
                        srcStart + i * srcPad,
                        srcStart + (i + 1) * srcPad,
                        dstStart + dstIndex * dstPad,
                        dstStart + (dstIndex + 1) * dstPad,
                        slice.Skip(1).ToArray(),
                        srcDim.Skip(1).ToArray(),
                        dstDim.Skip(1).ToArray(),
                        copytosrc);
                    dstIndex++;
                }
            }
            else
            {
                if (!copytosrc)
                {
                    Array.Copy(src, srcStart, dst, dstStart, dstEnd - dstStart);
                }
                else
                {
                    Array.Copy(dst, dstStart, src, srcStart,  dstEnd - dstStart);
                }
         
            }
        }


        private static int CalcPad(uint[] srcDim)
        {
            return (int)srcDim.Aggregate((long)1, (l, r) => l * r);
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


                var getshape = new Shape(new uint[] { 1} .Concat(Shape.Data.Skip(indexs.Length)).ToArray());


                var ret = new TOut();
                ret.Init(new Shape(
                    new uint[]{(uint) indexs[0].Length }  
                    .Concat(Shape.Data.Skip(indexs.Length)).ToArray()
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
                return Storage[retindex];
            }
            set
            {
                var retindex = GetIndexWithScalar(indexs);
                Storage[retindex] = value;
            }
        }



        private int GetIndexWithScalar(int[] indexs)
        {
            int retindex = 0;
            var srcDim = Shape.Data;
            for (int i = 0; i < indexs.Length; i++)
            {
                var index = indexs[i];
                var s = _slice[i];
                int srcPad = CalcPad(srcDim.Skip(i + 1).ToArray());

                index = s.Step > 0 ? s.Start + index : s.Start - index;
                retindex += (index*srcPad);
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
            ret.Init(new Shape(Shape.Size), Data);
            return ret;
        }

        public TOut Compare(NArray<T, TC, TOut> other)
        {
            if (Shape != other.Shape)
            {
                TOut retfalse = new TOut();
                retfalse.Init(new Shape(1), new T[] {(T) Convert.ChangeType(0, typeof(T))});
                return retfalse;
            }


            TOut ret = new TOut();
            ret.Init(Shape, this.Storage
                .Select((x, i) => Calculator.Compare(x, other.Storage[i])).ToArray());
            return ret;
        }


        public TOut ReShape(Shape shapeInput)
        {
            if (shapeInput.Size != Shape.Size)
            {
                throw new ArgumentException("size mistake");
            }
            TOut ret = new TOut();
            ret.Init(shapeInput, Data);
            return ret;
        }

        #region

        public T Sum()
        {
            return Calculator.Sum(Data);
        }

        private static void Argmax(NArray<T, TC, TOut> src,TOut dst, uint[] srcDim, int dimindex, int axis ,  List<Slice> srcSlice, List<int> dstIndex)
        {
            if (dimindex < srcDim.Length)
            {
                srcSlice.Add(0);
                if (axis == dimindex)
                {
                    srcSlice[dimindex] = ":";
                    Argmax(src, dst, srcDim, dimindex + 1, axis, srcSlice, dstIndex);
                }
                else
                {
                    dstIndex.Add(0);
                    var dstIndexLast = dstIndex.Count - 1;
                    var currentDim = srcDim[dimindex];

                    for (int i = 0; i < currentDim; i++)
                    {
                        srcSlice[dimindex] = i;
                        dstIndex[dstIndexLast] = i;
                        Argmax(src, dst, srcDim, dimindex + 1, axis, srcSlice, dstIndex);
                    }
                    dstIndex.RemoveAt(dstIndex.Count - 1);
                }
                srcSlice.RemoveAt(srcSlice.Count - 1);
            }
            else
            {
                dst[dstIndex.ToArray()] =(T)Convert.ChangeType( src[srcSlice.ToArray()].Argmax(),typeof(T));
            }

        }

        public int Argmax()
        {

            return Calculator.Argmax(Data);

        }

        public TOut Argmax(int axis )
        {
            var srcDim = Shape.Data.ToList();
            srcDim.RemoveAt(axis);
            TOut ret = new TOut();
            ret.Init(new Shape(srcDim.ToArray()));
            List<Slice> srcSlice = new List<Slice>();
            List<int> dstIndex = new List<int>();
            Argmax(this, ret, Shape.Data, 0, axis, srcSlice, dstIndex);
            return ret;
        }

        public TOut Log()
        {
            TOut ret = new TOut();
            ret.Init(Shape, Calculator.Log(Data));
            return ret;
        }

        public T Mean()
        {
            return Calculator.Mean(Data);
        }

        #endregion
    }
}
