import importlib

import Pyro4
import numpy as np
import unittestpythonwarp


@Pyro4.expose
class TestServer(object):
    def run(self, name, code, args):
        print("run %s" % name)
        importlib.reload(unittestpythonwarp)
        unit_test = unittestpythonwarp.UnitTestPythonWarp()
        result = getattr(unit_test, name)(code, **args)
        return result


def main():
    test_server = TestServer()
    daemon = Pyro4.Daemon()
    test_uri = daemon.register(test_server)
    ns = Pyro4.locateNS()
    ns.register("mxnet.csharp.testserver", test_uri)
    daemon.requestLoop()


if __name__ == "__main__":
    main()
