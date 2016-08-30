using System;

namespace mxnet.numerics.nbase
{
    public class NArrayStorage<T, TCalculatorC, TView> : NArray<T,TCalculatorC,TView>
        where T : new()
        where TCalculatorC : ICalculator<T>, new() 
        where TView : NArrayView<T, TCalculatorC, TView>, new()
    {
        public NArrayStorage(Shape shape)
        {
            Shape = new Shape(shape);
            Storage = new T[Shape.Size];
        }
        public NArrayStorage(Shape shape, T[] data)
        {
            Shape = new Shape(shape);
            Storage = new T[Shape.Size];
            Array.Copy(data, Storage, Math.Min(data.Length, Storage.Length));
        }
    }
}