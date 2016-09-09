using System;

namespace mxnet.csharp.initializer
{
    public abstract class Initializer
    {
        public void Call(string name, NdArray arr)
        {

            if (name.StartsWith("upsampling"))
            {
                this.InitBilinear(name, arr);
            }
            else if (name.StartsWith("stn_loc") && name.EndsWith("weight"))
            {
                this.InitZero(name, arr);
            }
            else if (name.StartsWith("stn_loc") && name.EndsWith("bias"))
            {
                this.InitLocBias(name, arr);
            }
            else if (name.EndsWith("bias"))
            {
                this.InitBias(name, arr);
            }
            else if (name.EndsWith("gamma"))
            {
                this.InitGamma(name, arr);
            }
            else if (name.EndsWith("beta"))
            {
                this.InitBeta(name, arr);
            }
            else if (name.EndsWith("weight"))
            {
                this.InitWeight(name, arr);
            }
            else if (name.EndsWith("moving_mean"))
            {
                this.InitZero(name, arr);
            }
            else if(name.EndsWith("moving_var"))
            {
                this.InitOne(name, arr);
            }
            else if (name.EndsWith("moving_inv_var"))
            {
                this.InitZero(name, arr);
            }
            else if (name.EndsWith("moving_avg"))
            {
                this.InitZero(name, arr);
            }
            else
            {
                this.InitDefault(name, arr);
            }
        }

        protected virtual void InitBilinear(string name, NdArray arr)
        {
            var shape = arr.GetShape().Data();
            var prodShape =Util.Prod(shape);
            float[] weight = new float[prodShape];

            var f = Math.Ceiling(shape[3] / 2.0);
            var c = (2 * f - 1 - f % 2) / (2.0 * f);
            for (int i = 0; i < prodShape; i++)
            {
                var x = i % shape[3];
                var y = (i / shape[3]) % shape[2];
                weight[i] = (float)((1 - Math.Abs(x / f - c)) * (1 - Math.Abs(y / f - c)));
            }

            arr.SyncCopyFromCpu(weight);
        }

        protected virtual void InitLocBias(string name, NdArray arr)
        {
            Util.Assert(arr.GetShape()[0] == 6);
            arr.SyncCopyFromCpu(new[] {1.0f, 0, 0, 0, 1.0f, 0});
        }

        protected virtual void InitBias(string name, NdArray arr)
        {
            arr.SetValue(0);
        }

        protected virtual void InitGamma(string name, NdArray arr)
        {
            arr.SetValue(1);
        }

        protected virtual void InitBeta(string name, NdArray arr)
        {
            arr.SetValue(0);
        }

        protected abstract void InitWeight(string name, NdArray arr);

        protected virtual void InitOne(string name, NdArray arr)
        {
            arr.SetValue(1);
        }

        protected virtual void InitZero(string name, NdArray arr)
        {
            arr.SetValue(0);
        }

        protected virtual void InitDefault(string name, NdArray arr)
        {
            throw new NotImplementedException($"Unknown initialization pattern for {name}");
        }
    }
}
