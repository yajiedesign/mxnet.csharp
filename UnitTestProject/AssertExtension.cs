using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject
{
   static class AssertExtension
    {
        public static T Throws<T>(Action expression_under_test,
                                     string exception_message = "Expected exception has not been thrown by target of invocation."
                                    ) where T : Exception
        {
            try
            {
                expression_under_test();
            }
            catch (T exception)
            {
                return exception;
            }

            Assert.Fail(exception_message);
            return null;
        }
    }
}
