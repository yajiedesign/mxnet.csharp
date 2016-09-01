using System;

namespace mxnet.csharp.initializer
{
    public class Uniform:Initializer
    {
        private readonly float scale;
        private readonly Random rand = new Random();
        public Uniform(float  scale)
        {
            this.scale = scale;
        }

        protected override void _init_weight(string name, NDArray arr)
        {
            //float[] temp = new float[arr.Size()];
            //for (int i = 0; i < temp.Length; i++)
            //{
            //    temp[i]= (float)_rand.NextDouble() % (_scale * 2) - _scale  ;
            //}

            //arr.SyncCopyFromCPU(temp);
            util.Random.Uniform(-scale, scale, arr);
        }
    }
}