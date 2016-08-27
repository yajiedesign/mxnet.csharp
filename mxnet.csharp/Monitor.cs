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
        private Func<NDArray, NDArray> stat_func;
        private int interval;
        private bool activated;
        private List<Tuple<int, string, NDArray>> queue;
        private int step;
        private List<Executor> exes;
        private Regex re_prog;
        private bool sort;


        NDArray asum_stat(NDArray x)
        {
            //TODO  return ndarray.norm(x)/Math.Sqrt(x.Size());
            return x;
        }

        public void stat_helper(string name, IntPtr array_ptr, IntPtr param2)
        {
            if (!this.activated || !this.re_prog.IsMatch(name))
            {
                return;
            }

            var array = new NDArray(array_ptr, writable: false);
            this.queue.Add(Tuple.Create(this.step, name, this.stat_func(array)));
        }
        public Monitor(int interval, Func<NDArray, NDArray> stat_func = null, string pattern = ".*", bool sort = false)
        {
            if (stat_func == null)
            {
                stat_func = asum_stat;
            }
            this.stat_func = stat_func;
            this.interval = interval;
            this.activated = false;
            this.queue = new List<Tuple<int, string, NDArray>>();
            this.step = 0;
            this.exes = new List<Executor>();
            this.re_prog = new Regex(pattern);
            this.sort = sort;



        }


        public void Install(Executor exe)
        {
            exe.set_monitor_callback(stat_helper);
            this.exes.Add(exe);
        }

        public void tic()
        {
            //TODO tic
        }

        public void toc()
        {
            //TODO toc
        }

        public void toc_print()
        {
            //TODO toc_print
        }
    }
}
