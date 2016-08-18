using System;
using System.Linq;
using System.Collections.Generic;

namespace opwrappergenerator
{
    internal class Op
    {
        private string name;
        private string description;
        private List<Arg> args;

        public Op(string name, string description, List<Arg> args)
        {
            this.name = name;
            this.description = description;

            var nameArg = new Arg(name,
                "symbol_name",
                "string",
                "name of the resulting symbol");
            args.Insert(0, nameArg);
            this.args = args.Where(w => !w.HasDefault).Concat(args.Where(w => w.HasDefault)).ToList();

        }
        /// <summary>
        /// 
        /// </summary>
        enum MyEnum
        { 
            
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="use_name"></param>
        /// <returns></returns>
        public string GetOpDefinitionString(bool use_name)
        {
            string ret = "";
            var argsLocal = this.args.Skip(use_name ? 0 : 1).ToList();

            if (name == "Reshape")
            {
                
            }

            //enum
            if (use_name)
            {
                foreach (var arg in argsLocal.Where(w => w.IsEnum))
                {
                    ret += $"/// <summary>\n/// {arg.Description.Replace("\n","")}\n/// </summary>\n";
                    ret += arg.Enum.GetDefinitionString();
                }
            }


            //comments 
            ret += $"/// <summary>\n/// {description.Replace("\n", "")}\n/// </summary>\n";


            foreach (var arg in argsLocal)
            {
                ret += $"/// <param name=\"{arg.Nane}\">{arg.Description.Replace("\n", "")}</param>";
            }
            ret += $" /// <returns>returns new symbol</returns>\n";


            ret += $"public Symbol {name}(";
            foreach (var arg in argsLocal)
            {
                ret += $"{arg.TypeName} {arg.Nane}";
                if (arg.HasDefault)
                {

                    ret += $"={arg.DefaultString}";
                }
                ret += ",\n";
            }
            if (argsLocal.Count > 0)
            {
                ret = ret.Substring(0, ret.Length - 2);
            }
      
            ret += ")\n{";

            foreach (var arg in argsLocal)
            {
                ret += arg.DefaultStringWithObject;
            }

            ret += $"return new Operator(\"{name}\")\n";

            foreach (var arg in args)
            {
                if (arg.TypeName == "Symbol" ||
                    arg.TypeName == "Symbol[]" ||
                    arg.TypeName == "string")
                {
                    continue;
                }
                ret += $".SetParam(\"{arg.Nane}\", {arg.Nane})\n";
            }


            foreach (var arg in args)
            {
                if (arg.TypeName != "Symbol")
                {
                    continue;
                }
                ret += $".SetInput(\"{arg.Nane}\", {arg.Nane})\n";
            }

            foreach (var arg in args)
            {
                if (arg.TypeName != "Symbol[]")
                {
                    continue;
                }
                ret += $".AddInput({arg.Nane})\n";
            }
            if (use_name)
            {
                ret += ".CreateSymbol(symbol_name);\n";
            }
            else
            {

                ret += ".CreateSymbol();\n";
            }
            ret += "}";
            return ret;
        }
    }
}