using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{

    public class NArray<T, TC, TView>
        where T : new()
        where TC : ICalculator<T>, new()
        where TView : NArrayView<T, TC, TView>, new()
    {
        private int _startIndex = 0;
        public Shape Shape { get; protected set; }

        protected T[] Storage;
  

        public IQueryable<T> Data => Storage?.AsQueryable().Skip(_startIndex).Take((int)Shape.Size);

        public GCHandle GetDataGcHandle()
        {
          return  GCHandle.Alloc(Storage, GCHandleType.Pinned);
        }

        private static readonly TC Calculator = new TC();

        public TView Flat()
        {
            var ret = new TView();
            ret.Shape = new Shape(Shape.Size);
            ret.Storage = Data.ToArray();
            return ret;
        }

        public TView Compare(NArray<T, TC, TView> other)
        {
            TView ret = new TView
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
