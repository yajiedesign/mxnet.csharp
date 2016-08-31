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

        private int _startIndex = 0;
        public Shape Shape { get; protected set; }

        protected T[] Storage;

        public T[] Data => Storage;

        public GCHandle GetDataGcHandle()
        {
          return  GCHandle.Alloc(Storage, GCHandleType.Pinned);
        }

        public NArray()
        {
            
        }

        public NArray(Shape shape)
        {
            Shape = new Shape(shape);
            Storage = new T[Shape.Size];
        }
        public NArray(Shape shape, T[] data)
        {
            Shape = new Shape(shape);
            Storage = new T[Shape.Size];
            Array.Copy(data, Storage, Math.Min(data.Length, Storage.Length));
        }


        public TOut this[Slice d1]
        {
            get
            {

                return null;
            }
        }

        public TOut this[int d0]
        {
            get
            {
                var ret = new TOut();
                ret.Shape = new Shape(Shape.Data.Skip(1).ToArray());
                ret.Storage = new T[ret.Shape.Size];

                int retindex = 0;
                int d1 =(int) Shape.Data[1];
                for (int i = 0; i < d1; i++)
                {
                    ret.Storage[retindex] = Storage[d0 * d1 + i];
                    retindex++;
                }
                return ret;
            }
        }

        public TOut Flat()
        {
            var ret = new TOut();
            ret.Shape = new Shape(Shape.Size);
            ret.Storage = Storage.ToArray();
            return ret;
        }

        public TOut Compare(NArray<T, TC, TOut> other)
        {
            TOut ret = new TOut
            {
                Shape = Shape,
                Storage = this.Storage
                    .Select((x, i) => Calculator.Compare(x, other.Storage[i])).ToArray()
  
            };
            return ret;
        }
        #region

        public T Sum()
        {
           return Calculator.Sum( this.Storage);
        }
        public int Argmax()
        {
            return Calculator.Argmax(this.Storage);
        }

        

        #endregion
    }
}
