using System;
using System.Linq;
using Random = mxnet.csharp.util.Random;

namespace mxnet.csharp.initializer
{
    public class Xavier:Initializer
    {
        private readonly RndType _rndType;
        private readonly FactorType _factorType;
        private readonly float _magnitude;

        public Xavier(RndType rndType = RndType.Uniform, FactorType factorType =  FactorType.Avg, float magnitude= 3)
        {
            this._rndType = rndType;
            this._factorType = factorType;
            this._magnitude = magnitude;
        }

        protected override void InitWeight(string name, NdArray arr)
        {
            var shape = arr.GetShape();
            float hwScale = 1.0f;
            if (shape.Size() > 2)
            {
                hwScale = Util.Prod(shape.Data().Skip(2).ToArray());
            }
            var fanOut = shape[0] * hwScale;
            var fanIn = shape[1] * hwScale;

            float factor = 1.0f;
            switch (_factorType)
            {
                case FactorType.Avg:
                    factor = (fanIn + fanOut)/2.0f;
                    break;
                case FactorType.In:
                    factor = fanIn;
                    break;
                case FactorType.Out:
                    factor = fanOut;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(FactorType));
            }

            var scale = (float)Math.Sqrt(this._magnitude/factor);

            switch (_rndType)
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
