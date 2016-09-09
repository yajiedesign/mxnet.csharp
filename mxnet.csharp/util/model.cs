using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using log4net;

namespace mxnet.csharp.util
{
    static class Model
    {
        public static void CheckArguments(Symbol symbol)
        {
            var argNames = symbol.ListArguments();
            var argNamesDuplicate = argNames.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));
            foreach (var name in argNamesDuplicate)
            {
                throw new Exception($"Find duplicated argument name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", argNames)}");
            }
            var auxNames = symbol.ListAuxiliaryStates();
            var auxNamesDuplicate = auxNames.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));

            foreach (var name in auxNamesDuplicate)
            {
                throw new Exception($"Find duplicated auxiliary name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", argNames)}");
            }
        }


        public static void SaveCheckpoint(string prefix,
            int? epoch,
            Symbol symbol,
            Dictionary<string, NdArray> argParams,
            Dictionary<string, NdArray> auxParams)
        {
            symbol.Save($"{prefix}-symbol.json");

            Dictionary<string, NdArray> dict = new Dictionary<string, NdArray>();
            foreach (var kv in argParams)
            {
                dict.Add(kv.Key, kv.Value);
            }
            foreach (var kv in auxParams)
            {
                dict.Add(kv.Key, kv.Value);
            }
            var paramName = $"{prefix}-{epoch:04d}.params";
            NdArray.Save(paramName, dict);
            ILog log = LogManager.GetLogger("");
            log.Info($"Saved checkpoint to \"{paramName}\"");
        }
    }
}
