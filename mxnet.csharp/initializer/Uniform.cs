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
            //float[] temp = new float[arr.Size()];
            //for (int i = 0; i < temp.Length; i++)
            //{
            //    temp[i]= (float)_rand.NextDouble() % (_scale * 2) - _scale  ;
            //}

            //arr.SyncCopyFromCPU(temp);
            util.Random.uniform(-_scale, _scale, arr);
        }
    }
}