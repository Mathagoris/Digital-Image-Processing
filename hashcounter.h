#ifndef HASHCOUNTER_H
#define HASHCOUNTER_H

#include <map>

class HashCounter
{
public:
    HashCounter() { m_map = std::map<int, int>(); }
    void increment(int key){
        if(m_map.find(key) == m_map.end()) {
            m_map[key] = 0;
        } else {
            m_map[key] = m_map[key] + 1;
        }
    }
    int operator [](int key){
        if(m_map.find(key) == m_map.end()) {
            return -1;
        } else {
            return m_map[key];
        }
    }
private:
    std::map<int,int> m_map;
};

#endif // HASHCOUNTER_H
