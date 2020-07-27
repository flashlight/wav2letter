#!/usr/bin/perl
use List::Util qw(min);

sub levenshtein
{
    my ($str1, $str2) = @_;
    my @ar1 = split ' ', $str1;
    my @ar2 = split ' ', $str2;

    my @dist;
    $dist[$_][0] = $_ foreach (0 .. @ar1);
    $dist[0][$_] = $_ foreach (0 .. @ar2);

    foreach my $i (1 .. @ar1) {
        foreach my $j (1 .. @ar2) {
            my $cost = $ar1[$i - 1] eq $ar2[$j - 1] ? 0 : 1;
            $dist[$i][$j] = min(
                $dist[$i - 1][$j] + 1,
                $dist[$i][$j - 1] + 1,
                $dist[$i - 1][$j - 1] + $cost
                );
        }
    }

    return $dist[@ar1][@ar2];
}

open(my $fh1, "$ARGV[0]");
open(my $fh2, "$ARGV[1]");
chomp(my @strings1=<$fh1>);
chomp(my @strings2=<$fh2>);

foreach my $str1 (@strings1) {
    foreach my $str2 (@strings2) {
        my $lev=levenshtein($str1, $str2);
        print "$str1|$str2|$lev\n";
        select()->flush();
    }
}
