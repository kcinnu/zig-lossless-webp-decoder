const std = @import("std");
const webp = @import("root.zig");

comptime {
    _ = @import("huffman_code.zig");
}

test {
    for (0..6) |case| {
        var path_buf: [256]u8 = undefined;
        const file = try std.fs.cwd().openFile(try std.fmt.bufPrint(&path_buf, "testcases/{}.webp", .{case}), .{});
        defer file.close();
        var file_buf = std.io.bufferedReader(file.reader());

        const actual = try webp.decode(std.testing.allocator, file_buf.reader());
        defer std.testing.allocator.free(actual.data);

        try std.testing.expectEqual(88, actual.width);
        try std.testing.expectEqual(31, actual.height);

        var expected_buf: [88 * 31 * 4]u8 = undefined;
        const expected = try std.fs.cwd().readFile(try std.fmt.bufPrint(&path_buf, "testcases/{}.rgba", .{case}), &expected_buf);

        for (actual.data, 0..) |a, i| {
            try std.testing.expectEqual(a.r, expected[i * 4 + 0]);
            try std.testing.expectEqual(a.g, expected[i * 4 + 1]);
            try std.testing.expectEqual(a.b, expected[i * 4 + 2]);
            try std.testing.expectEqual(a.a, expected[i * 4 + 3]);
        }
    }
}
