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
        private float _rescaleGrad;
        private readonly LrScheduler _lrScheduler;
        private readonly float _lr;
        private readonly float _wd;
        private Dictionary<string, float> _lrMult;
        private Dictionary<string, float> _wdMult;
        private readonly int _beginNumUpdate;
        private int _numUpdate;
        private readonly Dictionary<int, int> _indexUpdateCount;
        private float? _clipGradient;
        private readonly Dictionary<int, string> _idx2Name;
        private readonly Symbol _sym;


        public Optimizer(float rescaleGrad = 1.0f,
            Dictionary<int, string> paramIdx2Name = null,
            float wd = 0f,
            float? clipGradient = null,
            float learningRate = 0.01f,
            LrScheduler lrScheduler = null,
            Symbol sym = null,
            int beginNumUpdate = 0)
        {

            this._rescaleGrad = rescaleGrad;
            this._lr = learningRate;
            this._lrScheduler = lrScheduler;
            if (lrScheduler != null)
            {
                 this._lrScheduler.BaseLr = learningRate;
            }


            this._wd = wd;
            this._lrMult = new Dictionary<string, float>();
            this._wdMult = new Dictionary<string, float>();
            this._beginNumUpdate = beginNumUpdate;
            this._numUpdate = beginNumUpdate;
            this._indexUpdateCount = new Dictionary<int, int>();
            this._clipGradient = clipGradient;

            if (paramIdx2Name == null)
            {
                paramIdx2Name = new Dictionary<int, string>();
            }



            this._idx2Name = paramIdx2Name.ToDictionary(entry => entry.Key,
                                               entry => entry.Value);

            this._sym = sym;
            this.set_lr_mult(new Dictionary<string, float>());
            this.set_wd_mult(new Dictionary<string, float>());
        }
        private void set_lr_mult(Dictionary<string, float> argsLrMult)
        {
            this._lrMult = new Dictionary<string, float>();
            if (_sym != null)
            {
               var attr=  _sym.list_attr(true);
                foreach (var kv in attr)
                {
                    if (kv.Key.EndsWith("_lr_mult"))
                    {
                        _lrMult[kv.Key.Replace("_lr_mult", "")] = float.Parse(kv.Value);
                    }
                }
            }
            foreach (var kv in argsLrMult)
            {
                this._lrMult[kv.Key] = kv.Value;
            }
         
        }


        private void set_wd_mult(Dictionary<string, float> argsWdMult)
        {
            this._wdMult = new Dictionary<string, float>();
            foreach (var n in _idx2Name.Values)
            {
                if (!(n.EndsWith("_weight") || n.EndsWith("_gamma")))
                {
                    this._wdMult[n] = 0.0f;
                }
            }
            if (_sym != null)
            {
                var attr = _sym.list_attr(true);
                foreach (var kv in attr)
                {
                    if (kv.Key.EndsWith("_wd_mult"))
                    {
                        this._wdMult[kv.Key.Replace("_wd_mult", "")] = float.Parse(kv.Value);
                    }
                }
            }
            foreach (var kv in argsWdMult)
            {
                this._wdMult[kv.Key] = kv.Value;
            }

        }
        /// <summary>
        /// update num_update
        /// </summary>
        /// <param name="index">The index will be updated</param>
        public void _update_count(int index)
        {
            if (!this._indexUpdateCount.ContainsKey(index))
            {
                this._indexUpdateCount[index] = this._beginNumUpdate;
            }

            this._indexUpdateCount[index] += 1;
            this._numUpdate = Math.Max(this._indexUpdateCount[index], this._numUpdate);

        }


        public float _get_lr(int index)
        {
            float lr;
            if (this._lrScheduler != null)
            {
                lr = this._lrScheduler.Call(this._numUpdate);
            }

            else
            {
                lr = this._lr;
            }


            if (this._lrMult.ContainsKey(index.ToString()))
            {
                lr *= this._lrMult[index.ToString()];
            }
            else if(this._idx2Name.ContainsKey(index))
            {
                float v;
                if (!this._lrMult.TryGetValue(this._idx2Name[index], out v))
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
            if (this._wdMult.ContainsKey(index.ToString()))
            {
                wd *= this._wdMult[index.ToString()];
            }
            else if (this._idx2Name.ContainsKey(index))
            {
                float w;
                if (!this._wdMult.TryGetValue(this._idx2Name[index], out w))
                {
                    w = 1.0f;
                }
                wd *= w;
            }
            return wd;
        }


        public abstract NdArray create_state(int index , NdArray weight );

        public abstract void update(int index, NdArray weight, NdArray grad, NdArray state);






        private static void Update(Optimizer optimizer, int index, NdArray weight, NdArray grad, Dictionary<int, NdArray> states)
        {
            if (!states.ContainsKey(index))
            {
                states[index] = optimizer.create_state(index, weight);
            }

            optimizer.update(index, weight, grad, states[index]);
        }

        public static Action<int, NdArray, NdArray> GetUpdater(Optimizer optimizer)
        {
            Dictionary<int, NdArray> states = new Dictionary<int, NdArray>();

            return (int index, NdArray grad , NdArray weight) => { Update(optimizer, index, weight, grad, states); };
        }

        public string Serialize()
        {
            return "";

        }
        protected static OptimizerHandle _init_cc_optimizer(string name, string[] paramKeys, string[] paramVals)
        {
            IntPtr creator;
            Util.CallCheck(NativeMethods.MXOptimizerFindCreator(name,
                out creator));
            OptimizerHandle handle;
            Util.CallCheck(NativeMethods.MXOptimizerCreateOptimizer(
                creator,
                (uint) paramKeys.Count(),
                paramKeys, paramVals,
                out handle));

            return handle;
        }




        public static implicit operator Optimizer(string name)
        {
            var assemblies = AppDomain.CurrentDomain.GetAssemblies();

           var  optRegistry = assemblies.SelectMany(s1 => s1.Modules.SelectMany(s2 => s2.GetTypes()))
                .Where(w => w.IsSubclassOf(typeof(Optimizer) ))
                .ToDictionary(k=>k.Name.ToLower(),v=>v);



            if (optRegistry.ContainsKey(name.ToLower()))
            {
                var type = optRegistry[name.ToLower()];
                var constructors = type.GetConstructors();
                var con = constructors.First();
                var @params = con.GetParameters().Select(s => Type.Missing).ToArray();




                return (Optimizer) con.Invoke(@params);
            }

            return null;
        }


    }
}
