using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.initializer
{
    public class Xavier:Initializer
    {
        public enum factor_type

        {
            Avg,In,Out
        }
        public enum rnd_type

        {
            uniform,
            gaussian
        }

        private readonly rnd_type _rndType;
        private readonly factor_type _factorType;
        private readonly float _magnitude;

        public Xavier(rnd_type rnd_type = rnd_type.uniform, factor_type factor_type =  factor_type.Avg, float magnitude= 3)
        {
            _rndType = rnd_type;
            _factorType = factor_type;
            _magnitude = magnitude;
        }

        protected override void _init_weight(string name, NDArray arr)
        {
            var shape = arr.GetShape();
            float hw_scale = 1.0f;
            if (shape.Size() > 2)
            {
                hw_scale = Util.Prod(shape.data().Skip(2).ToArray());
            }
            var fan_out = shape[0] * hw_scale;
            var fan_in = shape[1] * hw_scale;

            float factor = 1.0f;
            switch (_factorType)
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
                    throw new ArgumentOutOfRangeException(nameof(factor_type));
            }

            var scale = (float)Math.Sqrt(this._magnitude/factor);

            switch (_rndType)
            {
                case rnd_type.uniform:
                    util.Random.uniform(-scale, scale, arr);
                    break;
                case rnd_type.gaussian:
                    util.Random.normal(-scale, scale, arr);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(rnd_type));
            }
        }
    }
}
