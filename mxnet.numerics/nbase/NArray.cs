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

        public Shape shape { get; protected set; }

        protected T[] storage;

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
            storage = new T[this.shape.size];
        }
        public NArray(Shape shape, T[] data)
        {
            this.shape = new Shape(shape);
            storage = new T[this.shape.size];
            Array.Copy(data, storage, Math.Min(data.Length, storage.Length));
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
            ret.storage = storage.ToArray();
            return ret;
        }

        public TOut Compare(NArray<T, TC, TOut> other)
        {
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
           return Calculator.Sum( this.storage);
        }
        public int Argmax()
        {
            return Calculator.Argmax(this.storage);
        }

        

        #endregion
    }
}
