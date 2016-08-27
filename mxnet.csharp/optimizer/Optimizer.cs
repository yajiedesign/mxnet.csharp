using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OptimizerCreator = System.IntPtr;
using OptimizerHandle = System.IntPtr;

namespace mxnet.csharp.optimizer
{
    public abstract class Optimizer
    {
        private float rescale_grad;
        private object lr_scheduler;
        private float lr;
        private float wd;
        private Dictionary<string, float> lr_mult;
        private Dictionary<object, object> wd_mult;
        private int begin_num_update;
        private int num_update;
        private Dictionary<object, object> _index_update_count;
        private object clip_gradient;
        private Dictionary<int, string> idx2name;
        private Symbol sym;

        public Optimizer(float rescale_grad = 1.0f, Dictionary<int, string> param_idx2name = null, float wd = 0f,
               object clip_gradient = null, float learning_rate = 0.01f,
               object lr_scheduler = null, Symbol sym = null, int begin_num_update = 0)
        {

            this.rescale_grad = rescale_grad;
            this.lr = learning_rate;
            this.lr_scheduler = lr_scheduler;
            if (lr_scheduler != null)
            {
                this.lr_scheduler.base_lr = learning_rate;
            }


            this.wd = wd;
            this.lr_mult = new Dictionary<string, float>();
            this.wd_mult = new Dictionary<object, object>();
            this.begin_num_update = begin_num_update;
            this.num_update = begin_num_update;
            this._index_update_count = new Dictionary<object, object>();
            this.clip_gradient = clip_gradient;

            if (param_idx2name == null)
            {
                param_idx2name = new Dictionary<int, string>();
            }



            this.idx2name = param_idx2name.ToDictionary(entry => entry.Key,
                                               entry => entry.Value);

            this.sym = sym;
            this.set_lr_mult(new Dictionary<string, float>());
            this.set_wd_mult(new Dictionary<object, object>());
        }
        private void set_lr_mult(Dictionary<string, float> args_lr_mult)
        {
            this.lr_mult = new Dictionary<string, float>();
            if (sym != null)
            {
               var attr=  sym.list_attr(true);
                foreach (var kv in attr)
                {
                    if (kv.Key.EndsWith("_lr_mult"))
                    {
                        lr_mult[kv.Key.Replace("_lr_mult", "")] = float.Parse(kv.Value);
                    }
                }
            }
            foreach (var kv in args_lr_mult)
            {
                this.lr_mult[kv.Key] = kv.Value;
            }
         
        }


        private void set_wd_mult(Dictionary<string, float> dictionary)
        {
            throw new NotImplementedException();
        }

     

        public abstract NDArray create_state(int index , NDArray weight );

        public abstract void update(int index, NDArray weight, NDArray grad, Dictionary<int, NDArray> state);



    }




    private static void Update(Optimizer optimizer, int index, NDArray weight, NDArray grad, Dictionary<int, NDArray> states)
        {
            if (!states.ContainsKey(index))
            {
                states[index] = optimizer.create_state(index, weight);
            }

            optimizer.update(index, weight, grad, states[index]);
        }

        public static Action<int, NDArray, NDArray> get_updater(Optimizer optimizer)
        {
            Dictionary<int, NDArray> states = new Dictionary<int, NDArray>();

            return (int index, NDArray weight, NDArray grad) => { Update(optimizer, index, weight, grad, states); };
        }
    }
}
