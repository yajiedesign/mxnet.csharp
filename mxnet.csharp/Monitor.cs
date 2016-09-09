using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public class Monitor
    {
        private readonly Func<NDArray, NDArray> _stat_func;
        private int _interval;
        private readonly bool _activated;
        private readonly List<Tuple<int, string, NDArray>> _queue;
        private readonly int _step;
        private readonly List<Executor> _exes;
        private readonly Regex _re_prog;
        private bool _sort;


        NDArray asum_stat(NDArray x)
        {
            //TODO  return ndarray.norm(x)/Math.Sqrt(x.size());
            return x;
        }

        public void stat_helper(string name, IntPtr array_ptr, IntPtr param2)
        {
            if (!this._activated || !this._re_prog.IsMatch(name))
            {
                return;
            }

            var array = new NDArray(array_ptr, writable: false);
            this._queue.Add(Tuple.Create(this._step, name, this._stat_func(array)));
        }
        public Monitor(int interval, Func<NDArray, NDArray> stat_func = null, string pattern = ".*", bool sort = false)
        {
            if (stat_func == null)
            {
                stat_func = asum_stat;
            }
            this._stat_func = stat_func;
            this._interval = interval;
            this._activated = false;
            this._queue = new List<Tuple<int, string, NDArray>>();
            this._step = 0;
            this._exes = new List<Executor>();
            this._re_prog = new Regex(pattern);
            this._sort = sort;



        }


        public void Install(Executor exe)
        {
            exe.set_monitor_callback(stat_helper);
            this._exes.Add(exe);
        }

        public void Tic()
        {
            //TODO tic
        }

        public void Toc()
        {
            //TODO toc
        }

        public void toc_print()
        {
            //TODO toc_print
        }
    }
}
