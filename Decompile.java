// Ghidra headless post-script: decompile functions by ADDRESS, write to a file.
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.program.model.address.Address;
import java.io.*;

public class Decompile extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] addrs = getScriptArgs();
        File outFile = new File("/tmp/ghidra_decomp.txt");
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);

        for (String a : addrs) {
            a = a.trim();
            if (a.isEmpty()) continue;
            Address addr;
            if (a.startsWith("0x") || a.startsWith("0X")) addr = toAddr(Long.parseLong(a.substring(2), 16));
            else addr = toAddr(Long.parseLong(a, 16));
            Function func = currentProgram.getFunctionManager().getFunctionAt(addr);
            if (func == null) {
                pw.println("=== NOT FOUND: " + a + " ===");
                continue;
            }
            pw.println("=== FUNCTION: " + func.getName() + " @ " + func.getEntryPoint()
                    + " size=" + func.getBody().getNumAddresses() + " ===");
            DecompileResults res = decomp.decompileFunction(func, 180, getMonitor());
            if (res != null && res.getDecompiledFunction() != null) {
                pw.println(res.getDecompiledFunction().getC());
            } else {
                pw.println("// decompilation failed or timed out");
            }
            pw.println("=== END " + func.getName() + " ===");
        }
        decomp.dispose();
        pw.close();
        println("WROTE: " + outFile.getAbsolutePath());
    }
}
