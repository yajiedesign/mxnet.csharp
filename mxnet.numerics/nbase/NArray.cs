using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{

    public class NArray<T,TC,TView> 
    where T : new()
    where TC : ICalculator<T>, new()
    where TView: NArrayView<T,TC, TView>, new()
    {
        public Shape Shape { get; protected set; }

        protected T[] Storage;

        public T[] Data => Storage;


        private static readonly TC Calculator = new TC();

        public TView Flat()
        {
            var ret = new TView();
            ret.Shape = new Shape(Shape.Size);
            ret.Storage = Data;
            return ret;
        }

        public NArray<T, TC, TView> Compare(NArray<T, TC, TView> other)
        {
            NArray<T, TC, TView> ret = new NArray<T, TC, TView>
            {
                Shape = Shape,
                Storage = this.Data
                    .Select((x, i) => Calculator.Compare(x, other.Storage[i]))
                    .ToArray()
            };
            return ret;
        }

    }
}
