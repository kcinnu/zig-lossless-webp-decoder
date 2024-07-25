const std = @import("std");

const webp = @import("root.zig");

pub fn main() !void {
    var heap = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = heap.allocator();
    defer _ = heap.detectLeaks();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    if (args.len < 3) {
        std.debug.print(
            \\you must specify an input and output path
            \\if youre running the binary directly:
            \\  ./path/to/decoder input.webp output.pam
            \\if youre using zig build run:
            \\  zig build run -- input.webp output.pam
        , .{});
        return;
    }

    const file = try std.fs.cwd().openFile(args[1], .{});
    defer file.close();

    var buf = std.io.bufferedReader(file.reader());

    const img = try webp.decode(alloc, buf.reader());
    defer alloc.free(img.data);

    const of = try std.fs.cwd().createFile(args[2], .{});
    defer of.close();
    try of.writer().print(
        \\P7
        \\WIDTH {}
        \\HEIGHT {}
        \\DEPTH 4
        \\MAXVAL 255
        \\TUPLTYPE RGB_ALPHA
        \\ENDHDR
        \\
    , .{ img.width, img.height });

    for (img.data) |*v| {
        v.* = @bitCast([4]u8{ v.r, v.g, v.b, v.a });
    }
    try of.writeAll(std.mem.sliceAsBytes(img.data));
}
