using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.initializer
{
    public abstract class Initializer
    {
        public void Call(string name, NDArray arr)
        {

            if (name.StartsWith("upsampling"))
            {
                this._init_bilinear(name, arr);
            }
            else if (name.StartsWith("stn_loc") && name.EndsWith("weight"))
            {
                this._init_zero(name, arr);
            }
            else if (name.StartsWith("stn_loc") && name.EndsWith("bias"))
            {
                this._init_loc_bias(name, arr);
            }
            else if (name.EndsWith("bias"))
            {
                this._init_bias(name, arr);
            }
            else if (name.EndsWith("gamma"))
            {
                this._init_gamma(name, arr);
            }
            else if (name.EndsWith("beta"))
            {
                this._init_beta(name, arr);
            }
            else if (name.EndsWith("weight"))
            {
                this._init_weight(name, arr);
            }
            else if (name.EndsWith("moving_mean"))
            {
                this._init_zero(name, arr);
            }
            else if(name.EndsWith("moving_var"))
            {
                this._init_one(name, arr);
            }
            else if (name.EndsWith("moving_inv_var"))
            {
                this._init_zero(name, arr);
            }
            else if (name.EndsWith("moving_avg"))
            {
                this._init_zero(name, arr);
            }
            else
            {
                this._init_default(name, arr);
            }
        }

        private void _init_bilinear(string name, NDArray arr)
        {
            var shape = arr.GetShape().data();
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

            arr.SyncCopyFromCPU(weight);
        }

        private void _init_loc_bias(string name, NDArray arr)
        {
            Debug.Assert(arr.GetShape()[0] == 6);
            arr.SyncCopyFromCPU(new float[] {1.0f, 0, 0, 0, 1.0f, 0});
        }

        private void _init_bias(string name, NDArray arr)
        {
            arr.SetValue(0);
        }

        private void _init_gamma(string name, NDArray arr)
        {
            arr.SetValue(1);
        }

        private void _init_beta(string name, NDArray arr)
        {
            arr.SetValue(0);
        }

        protected abstract void _init_weight(string name, NDArray arr);

        private void _init_one(string name, NDArray arr)
        {
            arr.SetValue(1);
        }

        private void _init_zero(string name, NDArray arr)
        {
            arr.SetValue(0);
        }

        private void _init_default(string name, NDArray arr)
        {
            throw new NotImplementedException($"Unknown initialization pattern for {name}");
        }
    }
}
