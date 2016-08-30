using System;

namespace mxnet.numerics.nbase
{
    public class NArrayStorage<T, TC, TView> : NArray<T,TC,TView>
        where T : new()
        where TC : ICalculator<T>, new() 
        where TView : NArrayView<T, TC, TView>, new()
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