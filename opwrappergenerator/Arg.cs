using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace opwrappergenerator
{
    class Arg
    {
        private Dictionary<string, string> typeDict = new Dictionary<string, string>
        {
            {"boolean", "bool"},
            {"Shape(tuple)", "Shape"},
            {"Symbol", "Symbol"},
            {"Symbol[]", "Symbol[]"},
            {"float", "float"},
            {"int", "int"},
            {"long", "long"},
            {"string", "string"}
        };

        public string Nane { get; }
        public string Description { get; }
        public bool IsEnum { get; } = false;
        public EnumType Enum { get; }
        public string TypeName { get; }
        public bool HasDefault { get; } = false;
        public string DefaultString { get; }

        public Arg(string opName = "", string argName = "", string typeString = "", string descString = "")
        {
            this.Nane = argName;
            this.Description = descString;
            if (typeString.StartsWith("{"))
            {
                IsEnum = true;
                Enum = new EnumType(opName + argName, typeString);
                TypeName = opName + argName;
            }
            else
            {
                string typename;
                if (typeDict.TryGetValue(typeString.Split(' ').First().Replace(",", ""), out typename))
                {
                    TypeName = typename;
                }
            }
            if (typeString.IndexOf("default", StringComparison.Ordinal) != -1)
            {
                HasDefault = true;
                DefaultString = typeString.Split(new string[] { "default=" }, StringSplitOptions.None)[1].Trim()
                    .Trim('\'');

                if (IsEnum)
                {
                    DefaultString = Enum.GetDefaultValueString(DefaultString);
                }
                else if (DefaultString == "False")
                {
                    DefaultString = "false";
                }
                else if (DefaultString == "True")
                {
                    DefaultString = "true";
                }
                else if (DefaultString.StartsWith("("))
                {
                    DefaultString = "Shape" + DefaultString;
                }
            }

        }
    }
    
}
