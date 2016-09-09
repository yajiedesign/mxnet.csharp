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
        public static T Throws<T>(Action expressionUnderTest,
                                     string exceptionMessage = "Expected exception has not been thrown by target of invocation."
                                    ) where T : Exception
        {
            try
            {
                expressionUnderTest();
            }
            catch (T exception)
            {
                return exception;
            }

            Assert.Fail(exceptionMessage);
            return null;
        }
    }
}
