using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.numerics.nbase
{
    public interface ICreateNArrayView<out TView, in TSrc>
    {
        TView Create(Shape shape, TSrc src);
    }

    public class NArrayView<T, TC, TView> : NArray<T, TC, TView>
        where T : new()
        where TC : ICalculator<T>, new()
        where TView : NArrayView<T, TC, TView>, new()
    {
        protected NArrayView()
        {

        }

        public NArrayView(Shape shape, NArray<T, TC, TView> src)
        {
            Shape = shape;
            Storage = src.Data;
        }
    }
}
