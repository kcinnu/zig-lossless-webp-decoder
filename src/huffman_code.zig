const std = @import("std");
const Code = @This();

len_cnt: [16]u16 = undefined,
map: []u16,
pub fn build(tmp: *Code, lens: []const u8) !void {
    @memset(&tmp.len_cnt, 0);
    var w_idx: u16 = 0;
    blk: {
        var cnt: u16 = 0;
        var idx: u16 = 0;
        for (lens, 0..) |l, i| {
            if (l > 1) break :blk;
            if (l == 1) {
                cnt += 1;
                idx = @intCast(i);
            }
        }
        if (cnt == 1) {
            tmp.len_cnt[0] = 1;
            tmp.map[0] = idx;
            return;
        }
    }
    var total_prob: u16 = 0;
    for (0..16, &tmp.len_cnt) |len2, *d| {
        if (len2 == 0) continue;
        var cnt: u16 = 0;
        for (lens, 0..) |len, idx| {
            if (len != len2) continue;
            total_prob += @as(u16, 1) << (15 - @as(u4, @intCast(len)));
            cnt += 1;
            tmp.map[w_idx] = @intCast(idx);
            w_idx += 1;
        }
        d.* = cnt;
    }
    if (total_prob != 1 << 15) return error.Invalid;
}

/// rd must be an std.io.BitReader
///
/// guaranteed to return values less than lens.len of the build call
pub fn readSym(code: *const Code, rd: anytype) !u16 {
    var acc: u16 = 0;
    var tmp: u16 = 0;
    for (code.len_cnt) |cnt| {
        if (acc < cnt) {
            return code.map[acc + tmp];
        }
        acc = ((acc - cnt) << 1) | (try rd.readBitsNoEof(u1, 1));
        tmp += cnt;
    }
    @panic("incomplete huffman code");
}

test {
    var buf: [4]u16 = undefined;
    var c = Code{ .map = &buf };
    try c.build(&.{ 1, 3, 3, 2 });
    var data: [2]u8 = undefined;
    std.mem.writeInt(u16, &data, 0b111011010, .little);
    var fbs = std.io.fixedBufferStream(&data);
    var rd = std.io.bitReader(.little, fbs.reader());
    try std.testing.expectEqual(0, try c.readSym(&rd));
    try std.testing.expectEqual(3, try c.readSym(&rd));
    try std.testing.expectEqual(1, try c.readSym(&rd));
    try std.testing.expectEqual(2, try c.readSym(&rd));
}
