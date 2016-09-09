using System;

namespace mxnet.csharp.initializer
{
    public class Uniform:Initializer
    {
        private readonly float _scale;
        private readonly Random _rand = new Random();
        public Uniform(float  scale)
        {
            this._scale = scale;
        }

        protected override void InitWeight(string name, NdArray arr)
        {
            //float[] temp = new float[arr.size()];
            //for (int i = 0; i < temp.Length; i++)
            //{
            //    temp[i]= (float)_rand.NextDouble() % (_scale * 2) - _scale  ;
            //}

            //arr.SyncCopyFromCPU(temp);
            util.Random.Uniform(-_scale, _scale, arr);
        }
    }
}