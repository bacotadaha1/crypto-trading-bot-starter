import inspect
import src.live as L

print("file:", getattr(L, "_file", "(no __file_)"))
print("_init_ in Trader:", "_init_" in inspect.getsource(L.Trader))

from src.live import Trader
t = Trader()
print("ok", hasattr(t, "ex"), getattr(t.ex, "id", "(no id)"))
