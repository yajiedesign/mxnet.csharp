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
  

        public IQueryable<T> Data => Storage?.AsQueryable().Skip(_startIndex).Take((int)Shape.Size);

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

        public TOut Flat()
        {
            var ret = new TOut();
            ret.Shape = new Shape(Shape.Size);
            ret.Storage = Data.ToArray();
            return ret;
        }

        public TOut Compare(NArray<T, TC, TOut> other)
        {
            TOut ret = new TOut
            {
                Shape = Shape,
                Storage = this.Data
                    .Select((x, i) => Calculator.Compare(x, other.Storage[i])).ToArray()
  
            };
            return ret;
        }

        public T Sum()
        {
           return Calculator.Sum( this.Data);
        }

    }
}
