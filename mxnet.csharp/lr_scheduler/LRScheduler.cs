using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;

namespace mxnet.csharp.lr_scheduler
{
    public abstract class LRScheduler
    {
        protected readonly ILog log;
        public float base_lr { get; set; } = 0.01f;

        protected LRScheduler(ILog log=null)
        {
            this.log = log ?? LogManager.GetLogger("");
        }

        /// <summary>
        ///  Call to schedule current learning rate
        /// 
        /// The training progress is presented by `num_update`, which can be roughly
        /// viewed as the number of minibatches executed so far. Its value is
        /// non-decreasing, and increases at most by one.
        /// 
        /// The exact value is the upper bound of the number of updates applied to
        /// a weight/index
        /// 
        /// See more details in https://github.com/dmlc/mxnet/issues/625
        /// </summary>
        /// <param name="num_update">the maximal number of updates applied to a weight.</param>
        public abstract float Call(int num_update);

    }
}


