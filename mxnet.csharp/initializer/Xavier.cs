using System;
using System.Linq;
using Random = mxnet.csharp.util.Random;

namespace mxnet.csharp.initializer
{
    public class Xavier:Initializer
    {
        private readonly RndType _rnd_type;
        private readonly FactorType _factor_type;
        private readonly float _magnitude;

        public Xavier(RndType rnd_type = RndType.Uniform, FactorType factor_type =  FactorType.Avg, float magnitude= 3)
        {
            this._rnd_type = rnd_type;
            this._factor_type = factor_type;
            this._magnitude = magnitude;
        }

        protected override void _init_weight(string name, NDArray arr)
        {
            var shape = arr.Get_shape();
            float hw_scale = 1.0f;
            if (shape.Size() > 2)
            {
                hw_scale = Util.Prod(shape.Data().Skip(2).ToArray());
            }
            var fan_out = shape[0] * hw_scale;
            var fan_in = shape[1] * hw_scale;

            float factor = 1.0f;
            switch (_factor_type)
            {
                case FactorType.Avg:
                    factor = (fan_in + fan_out)/2.0f;
                    break;
                case FactorType.In:
                    factor = fan_in;
                    break;
                case FactorType.Out:
                    factor = fan_out;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(FactorType));
            }

            var scale = (float)Math.Sqrt(this._magnitude/factor);

            switch (_rnd_type)
            {
                case RndType.Uniform:
                    Random.Uniform(-scale, scale, arr);
                    break;
                case RndType.Gaussian:
                    Random.Normal(-scale, scale, arr);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(RndType));
            }
        }
    }

    public enum RndType

    {
        Uniform,
        Gaussian
    }

    public enum FactorType

    {
        Avg,In,Out
    }
}
