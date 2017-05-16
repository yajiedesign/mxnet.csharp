using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp
{
    public interface INdArrayOrSymbol
    {
        IntPtr get_handle();
    }
}
