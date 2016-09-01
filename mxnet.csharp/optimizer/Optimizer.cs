using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mxnet.csharp.lr_scheduler;
using OptimizerCreator = System.IntPtr;
using OptimizerHandle = System.IntPtr;

namespace mxnet.csharp.optimizer
{


    [Serializable]
    public abstract class Optimizer
    {
        private float _rescale_grad;
        private readonly LRScheduler _lr_scheduler;
        private readonly float _lr;
        private readonly float _wd;
        private Dictionary<string, float> _lr_mult;
        private Dictionary<string, float> _wd_mult;
        private readonly int _begin_num_update;
        private int _num_update;
        private readonly Dictionary<int, int> _index_update_count;
        private float? _clip_gradient;
        private readonly Dictionary<int, string> _idx2_name;
        private readonly Symbol _sym;


        public Optimizer(float rescale_grad = 1.0f,
            Dictionary<int, string> param_idx2_name = null,
            float wd = 0f,
            float? clip_gradient = null,
            float learning_rate = 0.01f,
            LRScheduler lr_scheduler = null,
            Symbol sym = null,
            int begin_num_update = 0)
        {

            this._rescale_grad = rescale_grad;
            this._lr = learning_rate;
            this._lr_scheduler = lr_scheduler;
            if (lr_scheduler != null)
            {
                 this._lr_scheduler.base_lr = learning_rate;
            }


            this._wd = wd;
            this._lr_mult = new Dictionary<string, float>();
            this._wd_mult = new Dictionary<string, float>();
            this._begin_num_update = begin_num_update;
            this._num_update = begin_num_update;
            this._index_update_count = new Dictionary<int, int>();
            this._clip_gradient = clip_gradient;

            if (param_idx2_name == null)
            {
                param_idx2_name = new Dictionary<int, string>();
            }



            this._idx2_name = param_idx2_name.ToDictionary(entry => entry.Key,
                                               entry => entry.Value);

            this._sym = sym;
            this.set_lr_mult(new Dictionary<string, float>());
            this.set_wd_mult(new Dictionary<string, float>());
        }
        private void set_lr_mult(Dictionary<string, float> args_lr_mult)
        {
            this._lr_mult = new Dictionary<string, float>();
            if (_sym != null)
            {
               var attr=  _sym.list_attr(true);
                foreach (var kv in attr)
                {
                    if (kv.Key.EndsWith("_lr_mult"))
                    {
                        _lr_mult[kv.Key.Replace("_lr_mult", "")] = float.Parse(kv.Value);
                    }
                }
            }
            foreach (var kv in args_lr_mult)
            {
                this._lr_mult[kv.Key] = kv.Value;
            }
         
        }


        private void set_wd_mult(Dictionary<string, float> args_wd_mult)
        {
            this._wd_mult = new Dictionary<string, float>();
            foreach (var n in _idx2_name.Values)
            {
                if (!(n.EndsWith("_weight") || n.EndsWith("_gamma")))
                {
                    this._wd_mult[n] = 0.0f;
                }
            }
            if (_sym != null)
            {
                var attr = _sym.list_attr(true);
                foreach (var kv in attr)
                {
                    if (kv.Key.EndsWith("_wd_mult"))
                    {
                        this._wd_mult[kv.Key.Replace("_wd_mult", "")] = float.Parse(kv.Value);
                    }
                }
            }
            foreach (var kv in args_wd_mult)
            {
                this._wd_mult[kv.Key] = kv.Value;
            }

        }
        /// <summary>
        /// update num_update
        /// </summary>
        /// <param name="index">The index will be updated</param>
        public void _update_count(int index)
        {
            if (!this._index_update_count.ContainsKey(index))
            {
                this._index_update_count[index] = this._begin_num_update;
            }

            this._index_update_count[index] += 1;
            this._num_update = Math.Max(this._index_update_count[index], this._num_update);

        }


        public float _get_lr(int index)
        {
            float lr;
            if (this._lr_scheduler != null)
            {
                lr = this._lr_scheduler.Call(this._num_update);
            }

            else
            {
                lr = this._lr;
            }


            if (this._lr_mult.ContainsKey(index.ToString()))
            {
                lr *= this._lr_mult[index.ToString()];
            }
            else if(this._idx2_name.ContainsKey(index))
            {
                float v;
                if (!this._lr_mult.TryGetValue(this._idx2_name[index], out v))
                {
                    v = 1.0f;
                }

                lr *= v;
            }
            return lr;
        }


        public float _get_wd(int index)
        {
            var wd = this._wd;
            if (this._wd_mult.ContainsKey(index.ToString()))
            {
                wd *= this._wd_mult[index.ToString()];
            }
            else if (this._idx2_name.ContainsKey(index))
            {
                float w;
                if (!this._wd_mult.TryGetValue(this._idx2_name[index], out w))
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
            Util.CallCheck(NativeMethods.MXOptimizerFindCreator(name,
                out creator));
            OptimizerHandle handle;
            Util.CallCheck(NativeMethods.MXOptimizerCreateOptimizer(
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
}
