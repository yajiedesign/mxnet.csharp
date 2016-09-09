using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mxnet.csharp.util
{
    static class Model
    {
        public static void _check_arguments(Symbol symbol)
        {
            var arg_names = symbol.ListArguments();
            var arg_names_duplicate = arg_names.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));
            foreach (var name in arg_names_duplicate)
            {
                throw new Exception($"Find duplicated argument name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", arg_names)}");
            }
            var aux_names = symbol.ListAuxiliaryStates();
            var aux_names_duplicate = aux_names.GroupBy(i => i)
                .Where(g => g.Count() > 1)
                .Select(g => g.ElementAt(0));

            foreach (var name in aux_names_duplicate)
            {
                throw new Exception($"Find duplicated auxiliary name \"{name}\"," +
                                    $"please make the weight name non-duplicated(using name arguments)," +
                                    $"arguments are {String.Join(" ", arg_names)}");
            }
        }


        public static void save_checkpoint(string prefix,
            int? epoch,
            Symbol symbol,
            Dictionary<string, NDArray> arg_params,
            Dictionary<string, NDArray> aux_params)
        {
            symbol.Save($"{prefix}-symbol.json");


        }
    }
}
