class LocalBackend:
    def run(self, runner, until=None):
        return runner.execute(until)
