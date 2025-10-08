package org.example;

import java.util.Objects;
public class Edge {
    private final Integer start;
    private final Integer end;

    public Edge(Integer start,Integer end){
        this.start=start;
        this.end=end;
    }

    public Integer getStart() {
        return start;
    }

    public Integer getEnd() {
        return end;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Edge edge = (Edge) o;
        return Objects.equals(start, edge.start) &&
                Objects.equals(end, edge.end);
    }

    @Override
    public int hashCode() {
        return Objects.hash(start, end);
    }

    public String toString()
    {
        return start+" "+end;
    }
}
