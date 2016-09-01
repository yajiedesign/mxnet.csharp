using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OptimizerCreator = System.IntPtr;
using OptimizerHandle = System.IntPtr;

namespace mxnet.csharp.optimizer
{


    [Serializable]
    public abstract class Optimizer
    {
        private float rescale_grad;
        private Func<int,float> lr_scheduler;
        private float lr;
        private float wd;
        private Dictionary<string, float> lr_mult;
        private Dictionary<string, float> wd_mult;
        private int begin_num_update;
        private int num_update;
        private Dictionary<int, int> index_update_count;
        private float? clip_gradient;
        private Dictionary<int, string> idx2_name;
        private Symbol sym;


        public Optimizer(float rescale_grad = 1.0f,
            Dictionary<int, string> param_idx2name = null,
            float wd = 0f,
            float? clip_gradient = null,
            float learning_rate = 0.01f,
            Func<int, float> lr_scheduler = null,
            Symbol sym = null,
            int begin_num_update = 0)
        {

            this.rescale_grad = rescale_grad;
            this.lr = learning_rate;
            this.lr_scheduler = lr_scheduler;
            if (lr_scheduler != null)
            {
                //TODO   this.lr_scheduler.base_lr = learning_rate;
            }


            this.wd = wd;
            this.lr_mult = new Dictionary<string, float>();
            this.wd_mult = new Dictionary<string, float>();
            this.begin_num_update = begin_num_update;
            this.num_update = begin_num_update;
            this.index_update_count = new Dictionary<int, int>();
            this.clip_gradient = clip_gradient;

            if (param_idx2name == null)
            {
                param_idx2name = new Dictionary<int, string>();
            }



            this.idx2_name = param_idx2name.ToDictionary(entry => entry.Key,
                                               entry => entry.Value);

            this.sym = sym;
            this.set_lr_mult(new Dictionary<string, float>());
            this.set_wd_mult(new Dictionary<string, float>());
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


        private void set_wd_mult(Dictionary<string, float> args_wd_mult)
        {
            this.wd_mult = new Dictionary<string, float>();
            foreach (var n in idx2_name.Values)
            {
                if (!(n.EndsWith("_weight") || n.EndsWith("_gamma")))
                {
                    this.wd_mult[n] = 0.0f;
                }
            }
            if (sym != null)
            {
                var attr = sym.list_attr(true);
                foreach (var kv in attr)
                {
                    if (kv.Key.EndsWith("_wd_mult"))
                    {
                        this.wd_mult[kv.Key.Replace("_wd_mult", "")] = float.Parse(kv.Value);
                    }
                }
            }
            foreach (var kv in args_wd_mult)
            {
                this.wd_mult[kv.Key] = kv.Value;
            }

        }
        /// <summary>
        /// update num_update
        /// </summary>
        /// <param name="index">The index will be updated</param>
        public void _update_count(int index)
        {
            if (!this.index_update_count.ContainsKey(index))
            {
                this.index_update_count[index] = this.begin_num_update;
            }

            this.index_update_count[index] += 1;
            this.num_update = Math.Max(this.index_update_count[index], this.num_update);

        }


        public float _get_lr(int index)
        {
            float lr;
            if (this.lr_scheduler != null)
            {
                lr = this.lr_scheduler(this.num_update);
            }

            else
            {
                lr = this.lr;
            }


            if (this.lr_mult.ContainsKey(index.ToString()))
            {
                lr *= this.lr_mult[index.ToString()];
            }
            else if(this.idx2_name.ContainsKey(index))
            {
                float v;
                if (!this.lr_mult.TryGetValue(this.idx2_name[index], out v))
                {
                    v = 1.0f;
                }

                lr *= v;
            }
            return lr;
        }


        public float _get_wd(int index)
        {
            var wd = this.wd;
            if (this.wd_mult.ContainsKey(index.ToString()))
            {
                wd *= this.wd_mult[index.ToString()];
            }
            else if (this.idx2_name.ContainsKey(index))
            {
                float w;
                if (!this.wd_mult.TryGetValue(this.idx2_name[index], out w))
                {
                    w = 1.0f;
                }
                wd *= w;
            }
            return wd;
        }


        public abstract NDArray create_state(int index , NDArray weight );

        public abstract void update(int index, NDArray weight, NDArray grad, NDArray state);






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

            return (int index, NDArray grad , NDArray weight) => { Update(optimizer, index, weight, grad, states); };
        }

        public string Serialize()
        {
            return "";

        }
        protected static OptimizerHandle _init_cc_optimizer(string name, string[] param_keys, string[] param_vals)
        {
            IntPtr creator;
            Util.call_check(NativeMethods.MXOptimizerFindCreator(name,
                out creator));
            OptimizerHandle handle;
            Util.call_check(NativeMethods.MXOptimizerCreateOptimizer(
                creator,
                (uint) param_keys.Count(),
                param_keys, param_vals,
                out handle));

            return handle;
        }




        public static implicit operator Optimizer(string name)
        {
            var assemblies = AppDomain.CurrentDomain.GetAssemblies();

           var  opt_registry = assemblies.SelectMany(s1 => s1.Modules.SelectMany(s2 => s2.GetTypes()))
                .Where(w => w.IsSubclassOf(typeof(Optimizer) ))
                .ToDictionary(k=>k.Name.ToLower(),v=>v);



            if (opt_registry.ContainsKey(name.ToLower()))
            {
                var type = opt_registry[name.ToLower()];
                var constructors = type.GetConstructors();
                var con = constructors.First();
                var @params = con.GetParameters().Select(s => Type.Missing).ToArray();




                return (Optimizer) con.Invoke(@params);
            }

            return null;
        }


    }

    public class SGD : Optimizer
    {
        public override NDArray create_state(int index, NDArray weight)
        {
            throw new NotImplementedException();
        }

        public override void update(int index, NDArray weight, NDArray grad, NDArray state)
        {
            throw new NotImplementedException();
        }
    }

 
    public class CcSgd : Optimizer
    {
        private float momentum;
        private IntPtr handle;


        public CcSgd(float momentum = 0.0f,float rescale_grad = 1, Dictionary<int, string> param_idx2name = null, float wd = 0,
            float clip_gradient = -1, float learning_rate = 0.01F, Func<int, float> lr_scheduler = null,
            Symbol sym = null, int begin_num_update = 0)
            : base(rescale_grad, param_idx2name, wd, clip_gradient, learning_rate, lr_scheduler, sym, begin_num_update)
        {
            this.momentum = momentum;

            this.handle = Optimizer._init_cc_optimizer(
                "ccsgd",
                new[]
                {
                    "momentum",
                    "rescale_grad",
                    "clip_gradient"
                },
                new[]
                {
                    momentum.ToString(CultureInfo.InvariantCulture),
                    rescale_grad.ToString(CultureInfo.InvariantCulture),
                    clip_gradient.ToString(CultureInfo.InvariantCulture)
                });
        }

        public override NDArray create_state(int index, NDArray weight)
        {
            return null;
        }

        public override void update(int index, NDArray weight, NDArray grad, NDArray state)
        {
            var lr = this._get_lr(index);
            var wd = this._get_wd(index);
            this._update_count(index);
            Util.call_check(NativeMethods.MXOptimizerUpdate(this.handle,
                index,
                weight.Get_handle(),
                grad.Get_handle(),
                lr,
                wd));
        }
    }
}
