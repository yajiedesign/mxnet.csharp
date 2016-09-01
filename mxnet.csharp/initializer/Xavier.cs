using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.initializer
{
    public class xavier:Initializer
    {
        private readonly rnd_type rnd_type;
        private readonly factor_type factor_type;
        private readonly float magnitude;

        public xavier(rnd_type rnd_type = rnd_type.Uniform, factor_type factor_type =  factor_type.Avg, float magnitude= 3)
        {
            this.rnd_type = rnd_type;
            this.factor_type = factor_type;
            this.magnitude = magnitude;
        }

        protected override void _init_weight(string name, NDArray arr)
        {
            var shape = arr.GetShape();
            float hw_scale = 1.0f;
            if (shape.Size() > 2)
            {
                hw_scale = Util.prod(shape.data().Skip(2).ToArray());
            }
            var fan_out = shape[0] * hw_scale;
            var fan_in = shape[1] * hw_scale;

            float factor = 1.0f;
            switch (factor_type)
            {
                case factor_type.Avg:
                    factor = (fan_in + fan_out)/2.0f;
                    break;
                case factor_type.In:
                    factor = fan_in;
                    break;
                case factor_type.Out:
                    factor = fan_out;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(initializer.factor_type));
            }

            var scale = (float)Math.Sqrt(this.magnitude/factor);

            switch (rnd_type)
            {
                case rnd_type.Uniform:
                    util.Random.uniform(-scale, scale, arr);
                    break;
                case rnd_type.Gaussian:
                    util.Random.normal(-scale, scale, arr);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(initializer.rnd_type));
            }
        }
    }

    public enum rnd_type

    {
        Uniform,
        Gaussian
    }

    public enum factor_type

    {
        Avg,In,Out
    }
}
