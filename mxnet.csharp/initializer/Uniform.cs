using System;

namespace mxnet.csharp.initializer
{
    public class Uniform:Initializer
    {
        private readonly float _scale;
        private readonly Random _rand = new Random();
        public Uniform(float  scale)
        {
            _scale = scale;
        }

        protected override void _init_weight(string name, NDArray arr)
        {
            arr.SetValue((float) (_rand.NextDouble()%(_scale*2) - _scale));
        }
    }
}